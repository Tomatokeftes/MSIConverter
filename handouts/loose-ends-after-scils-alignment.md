# H. Loose ends after the SCiLS alignment

**Branch:** one per item, see each section (none created)
**Worktree:** `../Thyra-loose`
**Priority:** item 1 is not small and is not optional. The rest are chores.

```bash
git fetch origin && git worktree add -b <branch> ../Thyra-loose origin/main
```

Handout G (`resampling-scils-alignment.md`) is finished: items 1, 2, 3, 4, 6
and 7 all merged across v2.1.0-v2.2.3. This is what was found *around* that
work and deliberately not folded into it.

Everything below was measured on 2026-08-01 against `main` at **v2.2.3**
(`638e437`), Windows 11, Python 3.13, in the Poetry venv
`thyra-tfpMqqFS-py3.13`. **Do not re-derive it.** Where a claim is inference
rather than measurement, it says so.

---

## Not in this handout

Two larger questions are recorded elsewhere and are **not** chores. Do not
pick them up here without saying so:

- **Handout G item 5** -- a real conservative, intensity-splitting
  `tic_preserving`. Needs the owner's sign-off, because it changes stored
  values on the Rapiflex path.
- **The `linear_tof` gap.** No auto-detector ever returns `AxisType.LINEAR_TOF`,
  yet `bellini`'s grid fits p = 0.542 (robust fit over 38 log-spaced windows),
  i.e. the `linear_tof` law. On the axis it was being given, 67.4% of target
  bins never received a raw value and the axis was 5x too coarse above
  800 m/z. That is a design question about the detector chain, not a chore.

---

## 1. Ousia runs Thyra 2.2.3 against a cluster that violates three of its pins

**This is the item that matters.** It was found while checking whether the two
stored-values changes in v2.2.2 and v2.2.3 would reach Ousia. They already
have, and the pin situation is looser than "loose".

### What is actually installed

Measured in `<ousia-checkout>/backend/.venv`:

```
thyra code path : <thyra-checkout>/thyra/__init__.py
thyra version   : 2.2.3
spatialdata     : 0.7.0a1.dev62+gbfd2b5fac   thyra 2.x needs >=0.8.0, <0.9   VIOLATED
anndata         : 0.13.0.dev62+g3c90d3cc2    thyra 2.x needs >=0.13.2, <0.14 VIOLATED
zarr            : 3.2.1                      thyra 2.x needs >=3.0.0, <3.2   VIOLATED
dask            : 2026.6.0                   thyra 2.x needs >=2026.3.0, <2026.8  ok
```

Three of the four cluster pins are violated, silently.

### Why nothing caught it

`backend/pyproject.toml:31` declares the dependency as bare `"thyra"` with **no
version specifier at all**, and `[tool.uv.sources]` at `:115` resolves it to an
editable path dep:

```toml
thyra = { path = "../../Thyra", editable = true }
```

An editable path install is not re-resolved, so uv never applies Thyra's
declared floors. `backend/uv.lock:2762` still records `version = "1.25.4"` --
metadata from the 1.x era -- while the code being imported is whatever is
checked out in the sibling repo, currently `main` at 2.2.3.
`.venv/Lib/site-packages/thyra-1.25.4.dist-info` says the same stale thing.

Thyra's own `pyproject.toml:58-62` is explicit that this is dangerous:

> The spatialdata / ome-zarr / zarr / dask cluster is tightly coupled [...]
> without bounds pip resolves a mismatched set (a stale spatialdata alongside
> a much newer ome-zarr and dask), which fails to save/read back a [store]

That is the exact shape of what Ousia has, arrived at from the other
direction.

### Consequences, in order of how much they should worry you

1. **Ousia is running an unsupported combination right now**, and appears to
   work. Whether it works because it is fine or because nothing has exercised
   the incompatible path is **unknown and is the first thing to establish**.
2. **The v2.2.2 and v2.2.3 stored-values changes are already live in Ousia
   dev**, with nothing gating them. That is not necessarily wrong -- both are
   corrections -- but it happened without a decision.
3. **The next `uv sync` or re-lock is a cliff.** If the resolver ever applies
   Thyra 2.x's floors it must move spatialdata past 0.8 and anndata past
   0.13.2. Ousia pins both to *git forks* (`backend/pyproject.toml:102-103`)
   for reasons recorded in the scipy cap comment at `:41` -- the fork's
   backed-sparse reader calls scipy's `_validate_indices`. Moving them is not
   a version bump, it is a fork rebase.
4. **Packaging is aimed at the wrong pin.** The comment at
   `backend/pyproject.toml:112-114` says "when packaging the installer, swap
   this for a pinned PyPI version >=1.22". `>=1.22` is unbounded upward and
   would now resolve to 2.x, across a major boundary, into that same fork
   cluster.

### The work

**Do not start by changing a pin.** Start by establishing which of these is
true:

- (a) Thyra 2.x genuinely works on Ousia's fork cluster and its declared
  floors are too conservative -- in which case Thyra's floors are the thing
  to revisit, with evidence.
- (b) It does not, and Ousia is one untested code path away from a failure --
  in which case Ousia needs to either pin Thyra to the last 1.x that matches
  its cluster, or rebase its forks.

The cheapest discriminator: run Ousia's backend test suite, and a real
`convert_msi` + `read_lazy` round trip, in that venv against Thyra 2.2.3.
Thyra's own suite passes there, but Thyra's suite does not exercise Ousia's
pinned forks.

Then, whichever way it goes, **give the dependency a real constraint** so the
next person does not have to rediscover this. A bare `"thyra"` with an
editable source is indistinguishable from "any version, forever".

**Verify by making the claim fail first.** Before trusting a fix, confirm you
can reproduce the violation above with the one-liner in *Environment*.

---

## 2. `main` has no branch protection

**Status: BLOCKED 2026-08-03 -- the bypass this item assumed does not exist.**
Read "The trap" below before starting. The concurrency half of the release
gating shipped separately in #144; this half did not.

**Branch:** `chore/require-checks-on-main`

`gh api repos/M4i-Imaging-Mass-Spectrometry/thyra/branches/main/protection`
returns **404 "Branch not protected"**. There are no required status checks,
so nothing mechanically stops an untested PR from merging. During the handout
G merges, #130 merged four minutes after its first-ever checks appeared -- and
those only appeared because #127 merging caused it to be retargeted.

This was not worth fixing before, because PRs based on feature branches got
**zero** checks at all (a `pull_request` trigger's `branches:` filter matches
the *base*). PR #132 (`8ccca25`) fixed that. Required checks are now
meaningful, so turning them on is worth doing.

**Eleven** checks run today, not the seven this handout was written with.
PR #145 added the `integration` lane on 2026-08-03 and the list changed the
same day this item was first attempted. Re-enumerate from a recent **pull
request** before using it -- `main` only ever sees a subset:

```
clean-venv-install (3.12)      test (ubuntu-latest, 3.12)      integration (ubuntu-latest, 3.12)
clean-venv-install (3.13)      test (ubuntu-latest, 3.13)      integration (ubuntu-latest, 3.13)
complexity-check               test (windows-latest, 3.12)     integration (windows-latest, 3.12)
                               test (windows-latest, 3.13)     integration (windows-latest, 3.13)
```

Neither `tests.yml` nor `complexity-monitoring.yml` filters on `paths:`, so
every one of these runs on every PR. That matters: a required context that
does not run on some PR blocks that PR forever.

### The trap, and it is a real one -- now measured, and worse than described

**`release.yml` pushes directly to `main`.** `:119` runs `semantic-release
version`, which creates the version commit *and* the tag and pushes them, using
`secrets.GITHUB_TOKEN` (`:56`, `:123`). Branch protection with required status
checks will **block that push** unless the release actor can bypass it. A
release that cannot push leaves the repo tagged-but-not-bumped, or not tagged
at all.

(Line numbers are as of #144, which added the concurrency group. They have
moved twice already -- check before trusting them.)

Ousia has already been bitten by the adjacent version of this; see the
`release-pr-blocked-by-protection` note. Thyra's failure mode is different
(direct push, not a parked PR) but the cause is the same class.

**This was attempted on 2026-08-03 and the bypass could not be arranged.**
The plan above was to use a repository ruleset so the GitHub Actions app could
go in `bypass_actors`. That plan is **refuted**. Do not spend the afternoon on
it again.

Everything below was measured against a throwaway branch
(`ci/ruleset-probe-target`) with a real active ruleset and a workflow pushing
with `secrets.GITHUB_TOKEN` -- i.e. the exact release mechanism, not an
argument about it. All probe artefacts were deleted afterwards; the repo is
back to no protection and no rulesets.

| what pushed | bypass configured | result |
|---|---|---|
| `GITHUB_TOKEN` bot | none | **BLOCKED** -- `GH013 ... 7 of 7 required status checks are expected` |
| `GITHUB_TOKEN` bot | `RepositoryRole` 2 + 4 + 5 (write/maintain/admin) | **BLOCKED**, identical error |
| admin user's PAT | `RepositoryRole` 5 | **ALLOWED** |
| `GITHUB_TOKEN` bot | n/a, rules were `deletion` + `non_fast_forward` only | **ALLOWED** |

Two things follow.

**The Actions app cannot be a repo-level bypass actor at all.** Posting
`{"actor_id": 15368, "actor_type": "Integration"}` is rejected outright:

```
422 Validation Failed
Actor GitHub Actions integration must be part of the ruleset source or owner organization
```

`RepositoryRole`, `OrganizationAdmin` and `DeployKey` are all accepted -- only
the Actions app is not. And a repository role does not help, because the bot
holds an app-installation token and matches no collaborator role. That is the
third row of the table: the same bypass that lets a human admin through does
nothing for the bot.

**`tagged-but-not-bumped` is not hypothetical -- it is the default outcome.**
A `target: branch` ruleset does not govern tag refs. The probe pushed a tag
into the protected branch's repo while the branch push was being rejected, and
the tag **landed** (`PROBE_TAG_RESULT=ALLOWED`). So under required status
checks with no working bypass, `semantic-release version` gets its tag onto the
remote and loses the version commit. Exactly the state this handout feared.

### What is actually available

1. **A credential that is not `GITHUB_TOKEN`.** A PAT or deploy key belonging
   to an actor that *can* bypass, stored as a repo secret and used for
   `token:` in release.yml's checkout. Measured to work for an admin PAT with a
   `RepositoryRole: 5` bypass. The repo currently has **no secrets at all**
   (`actions/secrets` is empty), so this needs one minted first. Prefer a
   deploy key or a scoped GitHub App over a personal PAT -- a PAT ties releases
   to one person's account.
2. **An organization-level ruleset.** The 422 above says the app must belong to
   "the ruleset source or owner organization", which suggests an org ruleset
   would accept the Actions app where a repo ruleset will not. **Unverified** --
   it needs the `admin:org` scope, which the credentials on hand did not have.
   Verify before building on it.
3. **The safe subset, available right now.** `deletion` + `non_fast_forward`
   with no bypass at all is measured not to block the bot (fourth row). It stops
   main being force-pushed or deleted. It does *not* stop an untested merge, so
   it is not this item -- but it is free and it cannot break a release.

Until one of 1 or 2 is in place, **do not turn on required status checks for
`main`.** A repo that cannot cut a release is worse than one without required
checks.

Consider also requiring a linear history or not; that is a preference, not a
defect, and the repo currently uses merge commits.

---

## 3. `release.yml`'s "No release needed" branch is dead code

**Branch:** `ci/release-guard-honesty`

`release.yml:58-63`:

```bash
NEXT=$(semantic-release version --print 2>/dev/null || true)
if [ -z "$NEXT" ]; then
  echo "No release needed"
  echo "released=false" >> $GITHUB_OUTPUT
  exit 0
fi
```

`semantic-release version --print` returns the **current** version when nothing
is releasable, not an empty string. So `$NEXT` is never empty, `-z` never
fires, and that branch has never run. What actually stops a non-releasable
merge is the *tag* guard immediately below it at `:66-70`: the current version
is already tagged, so it takes the "already exists -- skipping" path.

It works, by accident, via a branch whose log message says something else.

The harm is diagnostic, not functional: a quiet release run reads as
"semantic-release decided not to bump" when it actually means "we tried to
re-release the current version and the tag guard caught it". That is
misleading in exactly the situation where you are trying to work out why a
release did not happen.

Fix the condition to test what it means -- compare `$NEXT` against the current
version, or drop the branch and let the tag guard own it and say so. Either
way the log line should describe what happened.

Cheap to confirm before changing anything: merge a `ci:`-typed commit (not
releasable) and read the job log. This has been observed but not isolated in a
throwaway run; treat the mechanism as measured and the fix as unverified.

---

## 4. Take spectrum representation explicitly, as SCiLS does

**Branch:** `feat/explicit-spectrum-type`

Handout G item 3 asked to *consider* this, and it was deliberately left out of
PR #127 because it adds user-facing API surface to a library with an external
consumer. That is the owner's call, which is why it is a separate item rather
than a follow-up commit.

SCiLS's importer takes representation as an argument (2026b User Guide p.81):

```
--rep_type arg   spectra representation (PROFILE, CENTROID)
```

After #127, `ImzMLMetadataExtractor._detect_centroid_spectrum` reads what the
file declares and only guesses when it declares nothing:

1. `_check_parser_metadata_for_spectrum_type` -- the fileDescription cvParams
2. `_check_xml_for_spectrum_type` -- the streaming `MS:1000127`/`MS:1000128` scan
3. `_guess_spectrum_type_from_storage_mode` -- the processed-mode guess, which
   now warns

An override belongs **ahead of step 1**, so a user can correct a file that
declares the wrong thing. The natural shape is a `spectrum_type` key in
`reader_options`, which `convert.py:107-108` already splats into the reader
constructor -- the same route `max_mass_axis_length` takes.

Open questions the implementer should answer rather than assume:

- **CLI flag, or `reader_options` only?** `max_mass_axis_length` is
  Python-API-only. SCiLS exposes `--rep_type` on the command line. Consistency
  argues both ways; pick one and say why.
- **Does it belong on the reader or the extractor?** The extractor is
  constructed by the reader (`imzml_reader.py:_create_metadata_extractor`), so
  the value has to be threaded through.
- **Should it warn when it contradicts the file?** Overriding a declared
  `MS:1000128` is a legitimate thing to want and also a good way to corrupt a
  conversion silently. A log line at minimum.

**This changes stored values for anyone who uses it**, by construction -- that
is the point of it. It changes nothing for anyone who does not.

---

## Constraints

**Do not undo handout G.** In particular: the PR #112 TIC rescale stays, the
Rapiflex default path measures exact (0.00000 ion-image error) and must not be
touched without a measurement showing improvement, and `tic_preserving` is
exact only when the source grid law equals the target axis law.

**`read_lazy` must keep working.** `anndata.experimental.read_lazy()` on Thyra
output is what Ousia depends on. Guarded by
`tests/unit/test_read_lazy_contract.py` and
`tests/unit/converters/test_lazy_loading_encoding.py`. Assert it still works,
not merely that files exist. Item 1 in particular is *about* that contract.

**Behaviour changes need before/after numbers in the PR.** Item 4 changes
output for some inputs. State plainly in each PR whether stored values change,
and for which inputs.

---

## Environment

**The venv trap (historical, now gone).** The numbers above were measured under
Poetry, where the venv was shared across worktrees and held `thyra` as an
editable install pointing at the *main* checkout, so `poetry run python` from a
worktree imported the wrong tree. `PYTHONPATH` alone was not enough --
`sys.path[0]`, the script's own directory, beat it -- so the workaround was to
call the venv interpreter by absolute path with `PYTHONPATH` set, and to assert
on `thyra.__file__` at the top of every standalone script.

uv removed the trap rather than renaming it. It resolves the project root by
walking up from the working directory, and a worktree has its own
`pyproject.toml`, so each worktree gets its own `.venv` holding its own source.
`uv run` syncs that environment before running, so from inside the worktree:

```bash
uv run pytest -q
```

**Reproducing item 1's violation**, from `<ousia-checkout>/backend`:

```bash
./.venv/Scripts/python.exe -c "
import thyra, spatialdata, anndata, zarr
print(thyra.__file__, thyra.__version__)
print(spatialdata.__version__, anndata.__version__, zarr.__version__)"
```

**Real data** lives outside every worktree, at
`<thyra-checkout>/test_data`: `bellini.imzML` (IONTOF TOF-SIMS,
profile, 16,384 spectra), `pea.imzML` (SCiLS/Bruker flex, centroid),
`20240826_xenium_0041899.imzML`, and two timsTOF `.d` folders. All five route
to `nearest_neighbor`.

**`caplog` captures nothing.** `setup_logging` sets `propagate = False` on the
`thyra` logger and other tests leave that state behind. Attach a handler to the
named logger instead -- `tests/unit/readers/test_bruker_region_selection.py`
has the pattern.

**Line endings.** Every commit touching a file fails the `mixed-line-ending`
hook once, rewrites to LF, and needs a second `git add` + `git commit`. Do not
run `pre-commit run --all-files`; it rewrites unrelated files.

**Releases are automatic.** `release.yml` runs on every push to `main`, so
**each merge cuts its own release** -- handout G's five PRs landed as five
separate versions. There is no batching window. A `ci:`- or `chore:`-typed
commit is not releasable and takes the tag-guard path (see item 3).

---

## Verification

```bash
uv run pytest -q
uv run black . && uv run isort . && uv run flake8
uv run --group docs mkdocs build --strict
```

Baseline on `main` at v2.2.3: **1049 passed, 11 skipped, 18 deselected**.

Beyond the suite:

- Item 1: Ousia's own backend suite in Ousia's venv, plus a real `convert_msi`
  + `read_lazy` round trip. Thyra's suite passing there proves nothing about
  Ousia's forks.
- Item 2: watch one real release complete end to end after enabling
  protection. Do not mark it done on configuration alone.
- Item 3: read an actual job log for a non-releasable merge.
- Item 4: convert `pea` and `bellini` end to end with and without the override
  and assert `read_lazy` opens both results.

## Deliverable

One PR per item; they are independent. Item 1 may well be a PR against
**Ousia**, not Thyra, or against both -- decide once you know which of (a) or
(b) is true, and say which in the PR.

State plainly in each PR whether stored values change, and for which inputs.
