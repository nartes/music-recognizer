PITCH_ESTIMATOR := python2 -B src/pitch-estimator.py

run-mic-test:
	$(PITCH_ESTIMATOR) mic

run-wav-test:
	$(PITCH_ESTIMATOR) wav tmp/sms-tools/sounds/piano.wav

debug-mic-test:
	$(PITCH_ESTIMATOR) mic debug

debug-wav-test:
	$(PITCH_ESTIMATOR) wav tmp/sms-tools/sounds/piano.wav debug

init-tmp-dir:
	mkdir tmp
	mkdir build

fetch-assets:
	cd tmp/; \
	git clone https://github.com/MTG/sms-tools.git

bulid-assets:
	cd tmp/sms-tools/software/models/utilFunctions_C; \
	python2 compileModule.py build_ext --inplace
