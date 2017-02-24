We would like to create an application that allows automatic music
transcription.

To start with something the dumb fourier transform based pitch estimator
is applied.

- [ ] Add dumb pitch estimator
- [ ] Add real time dumb pitch estimator with PyAudio
    - [ ] Local microphone
    - [ ] Web sockets
- [ ] Add NMF based real time dumb pitch estimator

# Install

```
make fetch-assets
make init-tmp-dir
```

# Run

```
make run-mic-test
```

or

```
make run-wav-test
```
