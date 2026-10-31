# `.github/workflows/`

GitHub Actions workflows. There is one: **`ci-cd.yaml`**, which tests and deploys the project.

## When it runs

Nothing runs when a pull request is opened. A run starts only after a merge, a tag or a manual trigger:

| Event | Environment | What happens |
|---|---|---|
| PR approved and merged into `dev` | dev | CI, then deploy |
| Tag `vX.Y.Z-rcN` pushed on a `dev` commit | rc | CI, then deploy; move the `rc` branch to the tag; open/update the PR `rc` → `main` |
| Release PR `rc` → `main` approved and merged | prod | CI; check that `main` equals a tested RC; deploy; publish GitHub release `vX.Y.Z` |
| Manual run (Actions → ci-cd → Run workflow) | chosen | redeploy or roll back (prod only from `main` or a `vX.Y.Z` tag) |

## Jobs

| Job | Purpose |
|---|---|
| `plan` | Works out the target environment and version, and runs the promotion guards |
| `ci` | `ruff` lint, `pytest` with coverage, compile all pipelines |
| `deploy` | Build/push the image, publish pipeline templates, start a training run, recreate schedules |
| `promote-rc` | RC only: point the `rc` branch at the tag and open/update the release PR |
| `release` | Prod only: create the `vX.Y.Z` tag and GitHub release |

`deploy` needs both `plan` and `ci`, so nothing is deployed unless the tests pass.

## Required settings

Create three GitHub environments, `dev`, `rc` and `prod`. In each, set:

| Name | Type |
|---|---|
| `GCP_PROJECT_ID` | variable |
| `WIF_PROVIDER` | secret |
| `DEPLOYER_SA` | secret |

`infra/setup_gcp.sh` prints these values. Branch protection and the other one-time steps are in the root README, section 7.
