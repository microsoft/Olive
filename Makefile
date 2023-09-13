WINDOWS                    ?= False
INSTALL_DEV_MODE           ?= False
EXAMPLE_FOLDER             ?=
EXAMPLE_NAME               ?=
INSTALL_EXTRAS             ?=
VERSION                    ?=
MODEL_NAME                 ?=
DEVICE                     ?=
ifeq ($(WINDOWS), True)
	CURRENT_DIR             = "$(subst /,\\,${CURDIR})"
	MKDIR_LOG_CMD           = mkdir logs | exit 0
	INSTALL_OLIVE_CMD       = "scripts\\install_olive.bat"
	TEST_CMD                = "scripts\\test.bat"
	TEST_EXAMPLES_CMD       = "scripts\\test_examples.bat"
	OVERWRITE_VERSION       = "python scripts\\overwrite_version.py --version $(VERSION)"
	PERF_CHECK_CMD          = "scripts\\run_performance_check.bat"
else
	CURRENT_DIR             = ${CURDIR}
	MKDIR_LOG_CMD           = mkdir -p logs
	INSTALL_OLIVE_CMD       = bash scripts/install_olive.sh
	TEST_CMD                = bash scripts/test.sh
	TEST_EXAMPLES_CMD       = bash scripts/test_examples.sh
	OVERWRITE_VERSION       = python scripts/overwrite_version.py --version $(VERSION)
	PERF_CHECK_CMD          = bash scripts/run_performance_check.sh
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
	$(INSTALL_OLIVE_CMD) $(INSTALL_DEV_MODE)

.PHONY: unit_test
unit_test:
	$(TEST_CMD) $(CURRENT_DIR) unit_test

.PHONY: integ_test
integ_test:
	$(TEST_CMD) $(CURRENT_DIR) integ_test

.PHONY: multiple_ep
multiple_ep:
	$(TEST_CMD) $(CURRENT_DIR) multiple_ep

.PHONY: test-examples
test-examples: logs/
test-examples:
	$(TEST_EXAMPLES_CMD) $(CURRENT_DIR) $(EXAMPLE_FOLDER) $(EXAMPLE_NAME)

.PHONY: performance
performance:
	$(PERF_CHECK_CMD) $(CURRENT_DIR) $(MODEL_NAME) $(DEVICE)

.PHONY: clean
clean:
	git clean -dfX
