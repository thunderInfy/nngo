.PHONY: check

check:
	go clean -testcache && go test -v ./...
