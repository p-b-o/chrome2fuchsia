CC      = cc
CFLAGS  = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lyajl -lm
TARGET  = c2f
SRC     = c2f.c
TESTDIR = tests
TESTS   = $(wildcard $(TESTDIR)/*.json)

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

test: $(TARGET)
	@pass=0; fail=0; \
	for f in $(TESTS); do \
		expected="$${f%.json}.fxt"; \
		./$(TARGET) < "$$f" > /tmp/c2f_out.fxt; \
		if cmp -s /tmp/c2f_out.fxt "$$expected"; then \
			echo "PASS: $$f"; pass=$$((pass+1)); \
		else \
			echo "FAIL: $$f"; fail=$$((fail+1)); \
		fi; \
	done; \
	echo "$$pass passed, $$fail failed"; \
	test $$fail -eq 0
