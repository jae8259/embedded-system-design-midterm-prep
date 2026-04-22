SHELL := /bin/bash

BUILD_SCRIPT := scripts/build_one.sh

BIN_DIR := bin

PROBLEMS := 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23

.PHONY: all clean gold $(addprefix p,$(PROBLEMS)) $(addprefix gold_p,$(PROBLEMS))
all: $(addprefix p,$(PROBLEMS))
gold: $(addprefix gold_p,$(PROBLEMS))

define PROBLEM_template
BIN_$(1) := $$(BIN_DIR)/p$(1)

p$(1):
	@"$$(BUILD_SCRIPT)" "$(1)" "$$(BIN_$(1))" mine
endef

define GOLD_template
GOLD_BIN_$(1) := $$(BIN_DIR)/gold_p$(1)

gold_p$(1):
	@"$$(BUILD_SCRIPT)" "$(1)" "$$(GOLD_BIN_$(1))" gold
endef

$(foreach p,$(PROBLEMS),$(eval $(call PROBLEM_template,$(p))))
$(foreach p,$(PROBLEMS),$(eval $(call GOLD_template,$(p))))

clean:
	rm -rf "$(BIN_DIR)"
