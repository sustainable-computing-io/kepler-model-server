### 0. Temporarily add `__init__.py` to all directories

```
find ./src -type d -exec touch {}/__init__.py \;
```

### 1. Generate `classes.plantuml` and `packages.plantuml` using the following commands
```
pyreverse --colorized --output plantuml --module-names y --show-stdlib --show-associated 2  --show-ancestors 1 --verbose -d umls/server/ --source-roots ./src/ ./src/server/
pyreverse --colorized --output plantuml --module-names y --show-stdlib --show-associated 2  --show-ancestors 1 --verbose -d umls/estimate/ --source-roots ./src/ ./src/estimate/
pyreverse --colorized --output plantuml --module-names y --show-stdlib --show-associated 2  --show-ancestors 1 --verbose -d umls/train/ --source-roots ./src/ ./src/train/
pyreverse --colorized --output plantuml --module-names y --show-stdlib --show-associated 2  --show-ancestors 1 --verbose -d umls/train/trainer/ --source-roots ./src/ ./src/train/trainer/
```

### 2. Use [plantuml](https://plantuml.com/download) to convert planuml files  to `svg` files
NeoVim plugin `neovim-soil` was used to generate svg files from plantuml files
