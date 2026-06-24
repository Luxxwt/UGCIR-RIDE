# ms-swift Patch

This project depends on the official `ms-swift` repository.

For the experiments in this repository, we use the official `ms-swift` codebase with a small patch applied to the embedding loss implementation.

## Tested upstream version

Official repository:

```text
https://github.com/modelscope/ms-swift
```

Tested commit:

```text
fb4be500c4fdb15be5b201f924a54dadf580eb7a
```

## Files modified by this patch

- `swift/loss/mapping.py`
- `swift/loss/embedding.py`

## Main changes

This patch:

- registers a new embedding loss type: `infonce_ranking`
- adds the corresponding `Infonce_ranking_Loss` implementation
- includes the ranking-based constraint used in our training setup
- preserves the remaining official `ms-swift` code structure

## How to apply

```bash
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
git checkout fb4be500c4fdb15be5b201f924a54dadf580eb7a
git apply /path/to/patches/ms_swift/0001-add-infonce-ranking-loss.patch
```

After applying the patch, point this repository's training scripts to your patched `ms-swift` checkout.
