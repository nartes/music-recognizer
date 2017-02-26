PITCH_ESTIMATOR := python2 -B src/pitch-estimator.py

npm-server:
	cd deps/browser-pcm-stream; npm start

run-mic-test:
	$(PITCH_ESTIMATOR) mic

run-wav-test:
	$(PITCH_ESTIMATOR) wav deps/sms-tools/sounds/piano.wav

run-websocket-test:
	$(PITCH_ESTIMATOR) websocket

debug-mic-test:
	$(PITCH_ESTIMATOR) mic debug

debug-wav-test:
	$(PITCH_ESTIMATOR) wav deps/sms-tools/sounds/piano.wav debug

debug-websocket-test:
	$(PITCH_ESTIMATOR) websocket debug

init-tmp-dir:
	mkdir tmp
	mkdir build
	git submodule init
	git submodule update

build-assets:
	cd deps/sms-tools/software/models/utilFunctions_C; \
	python2 compileModule.py build_ext --inplace
