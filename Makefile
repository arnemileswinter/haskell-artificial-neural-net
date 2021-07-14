out = out
data = $(out)/data
cblFile = haskell-aritificial-neural-net.cabal
stackBuildApp = stack build :$(@F) --profile --copy-bins --local-bin-path $(out)/

.PHONY: all
all: $(out)/xor $(out)/mnist

$(out)/xor: $(cblFile)
	$(stackBuildApp)

$(out)/mnist: $(data)/mnist-dataset $(cblFile) 
	$(stackBuildApp)

$(data)/mnist-dataset: $(data)
	$(shell cd $(data) && \
		mkdir mnist-dataset && \
		cd mnist-dataset && \
		wget https://data.deepai.org/mnist.zip && \
		unzip mnist.zip > /dev/null && \
		rm mnist.zip && \
		gunzip *.gz)

$(cblFile): package.yaml
	hpack

$(data): 
	mkdir $(data)

.PHONY: clean
clean:
	rm -r $(data)
	rm -r $(out)