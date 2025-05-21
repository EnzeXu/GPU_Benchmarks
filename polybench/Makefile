SUBDIRS := $(filter-out . .., $(wildcard */))
BENCHDIRS := $(foreach dir,$(SUBDIRS),$(if $(wildcard $(dir)Makefile),$(dir),))

.PHONY: all clean $(BENCHDIRS)

all: $(BENCHDIRS)
	@echo "All benchmarks built."

$(BENCHDIRS):
	@echo "Building in $@"
	@$(MAKE) -C $@

clean:
	@for dir in $(BENCHDIRS); do \
		$(MAKE) -C $$dir clean; \
	done
	@echo "All benchmarks cleaned."