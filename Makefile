WINDOWS                    ?= False
PIPELINE                   ?= False
INSTALL_DEV_MODE           ?= False
EXAMPLE                    ?=
INSTALL_EXTRAS             ?=
VERSION                    ?=
INSTALL_INDEX              ?=
ifeq ($(WINDOWS), True)
	CURRENT_DIR             = "$(subst /,\\,${CURDIR})"
	MKDIR_LOG_CMD           = mkdir logs | exit 0
	INSTALL_OLIVE_CMD       = "scripts\\install_olive.bat"
	TEST_CMD                = "scripts\\test.bat"
	TEST_EXAMPLES_CMD       = "scripts\\test_examples.bat"
	OVERWRITE_VERSION       = "python scripts\\overwrite_version.py --version $(VERSION)"
else
	CURRENT_DIR             = ${CURDIR}
	MKDIR_LOG_CMD           = mkdir -p logs
	INSTALL_OLIVE_CMD       = bash scripts/install_olive.sh
	TEST_CMD                = bash scripts/test.sh
	TEST_EXAMPLES_CMD       = bash scripts/test_examples.sh
	OVERWRITE_VERSION       = python scripts/overwrite_version.py --version $(VERSION)
endif

.PHONY: all
all:
	@echo "Please specify your command. Options: install-olive, test-examples, clean."

logs/:
	$(MKDIR_LOG_CMD)

.PHONY: overwrite-version
overwrite-version:
	$(OVERWRITE_VERSION)

.PHONY: install-olive
install-olive:
	$(INSTALL_OLIVE_CMD) $(PIPELINE) $(INSTALL_DEV_MODE)

.PHONY: test
test:
	$(TEST_CMD) $(PIPELINE) $(CURRENT_DIR)

.PHONY: test-examples
test-examples: logs/
test-examples:
	$(TEST_EXAMPLES_CMD) $(PIPELINE) $(CURRENT_DIR) $(EXAMPLE) $(INSTALL_INDEX)

.PHONY: clean
clean:
	git clean -dfX
