# AI-assisted development

Parts of this codebase, its tests, benchmark drivers and documentation were
written or revised with AI assistance: Anthropic Claude models through Claude
Code (including agentic sessions that implemented driver extensions, ran the
benchmark drivers on the maintainers' compute pool and drafted the resulting
documentation), and OpenAI GPT-5-series models through OpenCode. Pull requests
are reviewed by the maintainer and by automated code-review services (OpenAI
Codex and CodeRabbit). Every change is gated by the test suite and CI before it
reaches `main` (a pull request stacked on a parent branch gets that run when
its stack lands), and benchmark results carry metadata sidecars that pin the generating commit and
environment; the maintainers review and remain responsible for all code and
claims in this repository.

---

The mechanics behind that paragraph are documented in the rest of this section:
the gates are {doc}`ci`, what a measurement has to record is
{doc}`../benchmarks/index`, and what a contribution has to carry — whoever or
whatever drafted it — is {doc}`contributing`.

One scope note, since the statement above is deliberately kept as written. The
metadata guarantee is about **promoted** results — a full run wrapped by
tooling that pins the generating commit and the environment — not about every
run that happens to leave a file behind. The drivers those results came from
are no longer in this repository; the record a measurement still has to carry
is {doc}`../benchmarks/index`.
