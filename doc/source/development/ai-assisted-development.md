# AI-assisted development

Parts of this codebase, its tests, benchmark drivers and documentation were
written or revised with AI assistance: Anthropic Claude models through Claude
Code (including agentic sessions that implemented driver extensions, ran the
benchmark drivers on the maintainers' compute pool and drafted the resulting
documentation), and OpenAI GPT-5-series models through OpenCode. Pull requests
are reviewed by the maintainer and by automated code-review services (OpenAI
Codex and CodeRabbit). Every change is gated by the test suite and CI, and
benchmark results carry metadata sidecars that pin the generating commit and
environment; the maintainers review and remain responsible for all code and
claims in this repository.

---

The mechanics behind that paragraph are documented in the rest of this section:
the gates are {doc}`ci`, the metadata sidecars and the promotion rules are
{doc}`../benchmarks/index`, and what a contribution has to carry — whoever or
whatever drafted it — is {doc}`contributing`.

One scope note, since the statement above is deliberately kept as written. The
metadata guarantee is about **promoted** results — a full run wrapped by the
metadata tool, which is what pins the generating commit and the environment.
A raw or smoke-mode driver run is not covered: most drivers write no sidecar of
their own, only four do, and the suite's manifest records the commands rather
than the environment. {doc}`../benchmarks/index` has the per-driver detail and
the promotion sequence.
