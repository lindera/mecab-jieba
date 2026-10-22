VERSION := $(shell cat VERSION)
# The UD treebanks dict-src is trained on: GSD and the same sentences in
# simplified script (GSDSimp).
UD_GSD_COMMIT := 22f73e87f7ddc530bb131a6b7587971a03e1a712
UD_GSDSIMP_COMMIT := 4231dfd59866fa5999ad4a6bc1fdecd7985b3b59
EXPERIMENT ?= baseline
LAMBDA ?= 0.01
MAX_ITER ?= 100
REGULARIZATION ?= l2
ELASTIC_NET_L1_RATIO ?= 0.5

.DEFAULT_GOAL := help

help: ## Show help
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

clean: ## Clean the project
	rm -rf .mypy_cache
	rm -rf work

prepare: ## Prepare the project (download data, build dict, etc.)
	python3 scripts/build_jieba_csv.py
	mkdir -p work/ud-chinese
	git clone https://github.com/UniversalDependencies/UD_Chinese-GSD.git work/ud-chinese/UD_Chinese-GSD
	git -C work/ud-chinese/UD_Chinese-GSD checkout --quiet $(UD_GSD_COMMIT)
	git clone https://github.com/UniversalDependencies/UD_Chinese-GSDSimp.git work/ud-chinese/UD_Chinese-GSDSimp
	git -C work/ud-chinese/UD_Chinese-GSDSimp checkout --quiet $(UD_GSDSIMP_COMMIT)
	python3 scripts/build_seed.py
	python3 scripts/convert_conllu.py --input work/ud-chinese/UD_Chinese-GSD/ work/ud-chinese/UD_Chinese-GSDSimp/ --output work/train/corpus.txt --jieba-dict work/dict.txt.big --seed work/train/seed.csv --split train

train: ## Train CRF model and generate dict-src (output: work/experiments/$(EXPERIMENT)/dict-src/)
	LAMBDA=$(LAMBDA) MAX_ITER=$(MAX_ITER) REGULARIZATION=$(REGULARIZATION) ELASTIC_NET_L1_RATIO=$(ELASTIC_NET_L1_RATIO) bash scripts/run_experiment.sh $(EXPERIMENT)

tag: ## Make a new tag for the current version
	git tag $(VERSION)
	git push origin $(VERSION)
