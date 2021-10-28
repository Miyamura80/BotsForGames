install-npm:
	sudo apt install npm	

install-bazel: install-npm
	sudo npm install -g @bazel/bazelisk

