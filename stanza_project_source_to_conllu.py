#!/usr/bin/env python3
"""
Run Latvian Stanza on a normalized Latvian text, but STRICTLY keep sentence
boundaries already encoded in that normalized text by terminal punctuation.
Then project the analysis word-by-word onto a source wordform stream and write
CoNLL-U with source wordforms in FORM.

Usage:
    python3 stanza_project_source_to_conllu_strict_sents.py NORMALIZED.txt SOURCE.txt

Output:
    NORMALIZED.conllu

Main invariants:
  * one CoNLL-U token line is emitted for every whitespace token in SOURCE;
  * normalized and source inputs must have the same whitespace-token count;
  * sentence boundaries are not taken from Stanza's sentence splitter;
  * instead, boundaries are fixed before Stanza: after whitespace tokens ending
    in '.', '?', '!', or '…' (possibly followed by closing quotes/brackets);
  * each fixed sentence is passed separately to Stanza with tokenize_no_ssplit=True,
    so Stanza cannot merge it with the next sentence;
  * Unicode is written directly, without URL percent encoding such as %C4%81.

This is intended for workflows where the normalized text has already been
manually or procedurally sentence-punctuated and those boundaries must be kept.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class WTok:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class SentSpan:
    token_start: int          # inclusive, in the global whitespace-token stream
    token_end: int            # exclusive, in the global whitespace-token stream
    char_start: int           # inclusive, in normalized text
    char_end: int             # exclusive, in normalized text


@dataclass
class Row:
    global_idx: int
    source_form: str
    norm_form: str
    stanza_word: object
    stanza_pieces: List[object]
    conllu_id: int = 0


TERMINAL_RE = re.compile(r"[.!?…]+[\"'»”’\)]*$")


def read_text_without_optional_header(path: Path, header: str = "text") -> str:
    """Read UTF-8 text and drop a one-cell header 'text' if present."""
    s = path.read_text(encoding="utf-8-sig")
    lines = s.splitlines()
    if lines and lines[0].strip() == header:
        s = "\n".join(lines[1:])
    return s


def whitespace_tokens_with_offsets(text: str) -> List[WTok]:
    return [WTok(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def is_sentence_final_token(token: str, terminal_re: re.Pattern[str] = TERMINAL_RE) -> bool:
    """True when a whitespace token carries sentence-final punctuation.

    Examples:
      'nu.'       -> True
      'tur?'      -> True
      '[unint.]'  -> False, because the final character is ']'
      '[unint.].' -> True
      'ap[..]'    -> False
    """
    return bool(terminal_re.search(token))


def fixed_sentence_spans(norm_wtoks: Sequence[WTok]) -> List[SentSpan]:
    """Create fixed sentence spans from normalized whitespace tokens."""
    spans: List[SentSpan] = []
    start_i = 0

    for i, tok in enumerate(norm_wtoks):
        if is_sentence_final_token(tok.text):
            spans.append(
                SentSpan(
                    token_start=start_i,
                    token_end=i + 1,
                    char_start=norm_wtoks[start_i].start,
                    char_end=tok.end,
                )
            )
            start_i = i + 1

    if start_i < len(norm_wtoks):
        spans.append(
            SentSpan(
                token_start=start_i,
                token_end=len(norm_wtoks),
                char_start=norm_wtoks[start_i].start,
                char_end=norm_wtoks[-1].end,
            )
        )

    return spans


def token_index_for_span(tokens: Sequence[WTok], start: Optional[int], end: Optional[int]) -> Optional[int]:
    """Return the local whitespace-token index containing a Stanza word span."""
    if start is None or end is None:
        return None
    for i, tok in enumerate(tokens):
        if tok.start <= start and end <= tok.end:
            return i
        # Some tokenizers can report end on attached punctuation just after a word.
        if tok.start <= start < tok.end and end == tok.end + 1:
            return i
    return None


def get_word_span(word) -> Tuple[Optional[int], Optional[int]]:
    start = getattr(word, "start_char", None)
    end = getattr(word, "end_char", None)
    if start is not None and end is not None:
        return start, end
    parent = getattr(word, "parent", None)
    if parent is not None:
        return getattr(parent, "start_char", None), getattr(parent, "end_char", None)
    return None, None


def has_word_char(s: str) -> bool:
    """True if the token contains at least one Unicode letter or digit."""
    return any(ch.isalnum() for ch in s)


def choose_analysis_word(pieces: List[object]) -> object:
    """Pick one Stanza word as the analysis carrier for a whitespace token.

    Prefer a piece containing letters/digits, even when Stanza labels it PUNCT;
    this keeps discourse items such as 'mhm'. Among such pieces, prefer a
    non-PUNCT UPOS and then the longest surface form.
    """
    if not pieces:
        raise ValueError("empty Stanza piece group")

    alpha = [w for w in pieces if has_word_char(getattr(w, "text", "") or "")]
    if alpha:
        non_punct = [w for w in alpha if getattr(w, "upos", None) != "PUNCT"]
        pool = non_punct or alpha
        return max(pool, key=lambda w: len(getattr(w, "text", "") or ""))

    return max(pieces, key=lambda w: len(getattr(w, "text", "") or ""))


def escape_misc_value(value: str) -> str:
    """Return a CoNLL-U MISC-safe value without URL percent-encoding."""
    return (
        str(value)
        .replace("\t", "_")
        .replace("\n", "_")
        .replace("\r", "_")
        .replace(" ", "_")
        .replace("|", "¦")
    )


def merge_misc(*parts: Optional[str]) -> str:
    vals: List[str] = []
    for p in parts:
        if not p or p == "_":
            continue
        vals.extend(x for x in str(p).split("|") if x)
    return "|".join(vals) if vals else "_"


def conllu_field(value) -> str:
    if value is None or value == "":
        return "_"
    return str(value)


def safe_field(value: str) -> str:
    return value.replace("\t", " ").replace("\n", " ").replace("\r", " ")


def build_pipeline(processors: str, download: bool, use_gpu: bool):
    try:
        import stanza
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Stanza is not installed. Install it with:\n"
            "    pip install stanza\n"
            "Then download Latvian models once with:\n"
            "    python -c \"import stanza; stanza.download('lv')\""
        ) from exc

    def make_pipeline():
        return stanza.Pipeline(
            lang="lv",
            processors=processors,
            tokenize_no_ssplit=True,
            use_gpu=use_gpu,
            verbose=False,
        )

    try:
        return make_pipeline()
    except Exception as exc:
        if not download:
            raise SystemExit(
                "Could not initialize Latvian Stanza pipeline. Run once:\n"
                "    python -c \"import stanza; stanza.download('lv')\"\n"
                "or rerun this script with --download.\n\n"
                f"Original error: {exc}"
            ) from exc
        stanza.download("lv")
        return make_pipeline()


def stanza_piece_groups(sentence, local_wtoks: Sequence[WTok]) -> Dict[int, List[object]]:
    """Map local whitespace-token index -> Stanza word pieces."""
    groups: Dict[int, List[object]] = {}
    for word in sentence.words:
        start, end = get_word_span(word)
        idx = token_index_for_span(local_wtoks, start, end)
        if idx is not None:
            groups.setdefault(idx, []).append(word)
    return groups


def project_one_fixed_sentence(
    *,
    nlp,
    sent_no: int,
    span: SentSpan,
    norm_text: str,
    norm_wtoks_global: Sequence[WTok],
    src_wtoks_global: Sequence[WTok],
) -> List[str]:
    """Analyze one fixed sentence with Stanza and project it to source FORM."""
    seg_text = norm_text[span.char_start : span.char_end]
    local_norm_wtoks = whitespace_tokens_with_offsets(seg_text)
    expected_len = span.token_end - span.token_start

    if len(local_norm_wtoks) != expected_len:
        raise SystemExit(
            f"Internal sentence-span error at sentence {sent_no}:\n"
            f"  expected local tokens: {expected_len}\n"
            f"  found local tokens:    {len(local_norm_wtoks)}\n"
            f"  segment: {seg_text!r}"
        )

    doc = nlp(seg_text)
    # tokenize_no_ssplit=True should give one sentence, but even if a model still
    # returns more, merge the word pieces into this fixed sentence instead of
    # emitting extra sentence boundaries.
    all_words: List[object] = []
    for st_sent in doc.sentences:
        all_words.extend(st_sent.words)

    class MergedSentence:
        words = all_words

    groups = stanza_piece_groups(MergedSentence, local_norm_wtoks)

    missing_local = [i for i in range(expected_len) if i not in groups]
    if missing_local:
        first = missing_local[0]
        context_start = max(0, first - 5)
        context_end = min(expected_len, first + 6)
        ctx = "\n".join(
            f"  global {span.token_start + i + 1}: "
            f"norm={norm_wtoks_global[span.token_start + i].text!r} | "
            f"src={src_wtoks_global[span.token_start + i].text!r}"
            for i in range(context_start, context_end)
        )
        raise SystemExit(
            f"Word alignment failed inside fixed sentence {sent_no}.\n"
            f"  fixed sentence tokens: {expected_len}\n"
            f"  missing local token numbers: "
            f"{', '.join(str(i + 1) for i in missing_local[:20])}\n"
            f"  first suspicious area:\n{ctx}"
        )

    rows: List[Row] = []
    for local_i in range(expected_len):
        global_i = span.token_start + local_i
        pieces = groups[local_i]
        chosen = choose_analysis_word(pieces)
        rows.append(
            Row(
                global_idx=global_i,
                source_form=src_wtoks_global[global_i].text,
                norm_form=norm_wtoks_global[global_i].text,
                stanza_word=chosen,
                stanza_pieces=pieces,
                conllu_id=local_i + 1,
            )
        )

    out_lines: List[str] = []
    out_lines.append(f"# sent_id = {sent_no}")
    out_lines.append(f"# fixed_sentence = punct")
    out_lines.append("# text = " + " ".join(r.source_form for r in rows))
    out_lines.append("# norm_text = " + " ".join(r.norm_form for r in rows))

    # Stanza HEADs are local to its sentence. Since we merge any accidental Stanza sub-sentences back into the fixed sentence, remapping is exact for
    # ordinary one-sentence output and conservative otherwise.
    old_to_new: Dict[int, int] = {}
    for r in rows:
        try:
            old_to_new[int(getattr(r.stanza_word, "id"))] = r.conllu_id
        except Exception:
            pass

    # If Stanza accidentally produced several internal sentences, word ids may
    # repeat. In that rare case old_to_new is ambiguous; better keep roots than
    # invent cross-sentence dependencies.
    repeated_word_ids = len(old_to_new) < len(rows)

    def remap_head(word) -> int:
        if repeated_word_ids:
            return 0 if int(getattr(word, "head", 0) or 0) == 0 else 0
        head = int(getattr(word, "head", 0) or 0)
        if head == 0:
            return 0
        if head in old_to_new:
            return old_to_new[head]
        seen = set()
        while head and head not in old_to_new and head not in seen:
            seen.add(head)
            try:
                parent = all_words[head - 1]
                head = int(getattr(parent, "head", 0) or 0)
            except Exception:
                return 0
        return old_to_new.get(head, 0)

    for r in rows:
        w = r.stanza_word
        piece_texts = [getattr(p, "text", "") or "" for p in r.stanza_pieces]
        misc = merge_misc(
            getattr(w, "misc", None),
            f"NormWord={escape_misc_value(getattr(w, 'text', '') or '')}",
            f"NormTok={escape_misc_value(r.norm_form)}",
            f"SourceIndex={r.global_idx + 1}",
            f"FixedSent={sent_no}",
            (
                f"StanzaPieces={escape_misc_value('+'.join(piece_texts))}"
                if len(piece_texts) > 1 else None
            ),
        )

        out_lines.append(
            "\t".join(
                [
                    str(r.conllu_id),
                    safe_field(r.source_form),
                    conllu_field(getattr(w, "lemma", None)),
                    conllu_field(getattr(w, "upos", None)),
                    conllu_field(getattr(w, "xpos", None)),
                    conllu_field(getattr(w, "feats", None)),
                    str(remap_head(w)),
                    conllu_field(getattr(w, "deprel", None)),
                    "_",
                    misc,
                ]
            )
        )

    return out_lines


def project_to_conllu_strict_sentences(nlp, norm_text: str, norm_wtoks: Sequence[WTok], src_wtoks: Sequence[WTok]) -> Tuple[str, int]:
    if len(norm_wtoks) != len(src_wtoks):
        raise SystemExit(
            "Whitespace-token counts differ before Stanza alignment:\n"
            f"  normalized: {len(norm_wtoks)}\n"
            f"  source:     {len(src_wtoks)}\n"
            "The two input files must contain the same number of whitespace tokens."
        )

    spans = fixed_sentence_spans(norm_wtoks)
    if not spans:
        raise SystemExit("No fixed sentence spans found.")

    out_lines: List[str] = []
    emitted = 0
    for sent_no, span in enumerate(spans, start=1):
        sent_lines = project_one_fixed_sentence(
            nlp=nlp,
            sent_no=sent_no,
            span=span,
            norm_text=norm_text,
            norm_wtoks_global=norm_wtoks,
            src_wtoks_global=src_wtoks,
        )
        out_lines.extend(sent_lines)
        out_lines.append("")
        emitted += span.token_end - span.token_start

    if emitted != len(src_wtoks):
        raise SystemExit(
            "Internal error: fixed sentence spans do not cover the source stream exactly.\n"
            f"  emitted:  {emitted}\n"
            f"  expected: {len(src_wtoks)}"
        )

    return "\n".join(out_lines).rstrip() + "\n", len(spans)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run Latvian Stanza on normalized text while strictly preserving "
            "sentence boundaries already marked by punctuation, then project "
            "the analysis onto source wordforms."
        )
    )
    p.add_argument("normalized", type=Path, help="normalized Latvian text; sentence punctuation is authoritative")
    p.add_argument("source", type=Path, help="source wordform stream with the same whitespace-token count")
    p.add_argument("-o", "--output", type=Path, default=None, help="output .conllu; default: NORMALIZED.conllu")
    p.add_argument("--processors", default="tokenize,pos,lemma,depparse", help="Stanza processors")
    p.add_argument("--download", action="store_true", help="download Latvian Stanza models if needed")
    p.add_argument("--gpu", action="store_true", help="use GPU if available")
    p.add_argument(
        "--check-sentences-only",
        action="store_true",
        help="only print the fixed sentence count and token count; do not run Stanza",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    norm_text = read_text_without_optional_header(args.normalized)
    src_text = read_text_without_optional_header(args.source)
    norm_wtoks = whitespace_tokens_with_offsets(norm_text)
    src_wtoks = whitespace_tokens_with_offsets(src_text)

    if not norm_wtoks:
        raise SystemExit(f"No whitespace tokens found in {args.normalized}")
    if not src_wtoks:
        raise SystemExit(f"No whitespace tokens found in {args.source}")
    if len(norm_wtoks) != len(src_wtoks):
        raise SystemExit(
            "The two inputs must have the same number of whitespace tokens before Stanza.\n"
            f"  {args.normalized}: {len(norm_wtoks)}\n"
            f"  {args.source}:     {len(src_wtoks)}"
        )

    spans = fixed_sentence_spans(norm_wtoks)
    if args.check_sentences_only:
        print(f"Whitespace words: {len(norm_wtoks)}")
        print(f"Fixed sentences:  {len(spans)}")
        return 0

    nlp = build_pipeline(args.processors, download=args.download, use_gpu=args.gpu)
    conllu, sent_count = project_to_conllu_strict_sentences(nlp, norm_text, norm_wtoks, src_wtoks)

    out_path = args.output if args.output is not None else args.normalized.with_suffix(".conllu")
    out_path.write_text(conllu, encoding="utf-8")

    print(f"Wrote: {out_path}", file=sys.stderr)
    print(f"Whitespace words projected: {len(src_wtoks)}", file=sys.stderr)
    print(f"Fixed sentences preserved: {sent_count}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
