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
make init-tmp-dir
make fetch-assets
make build-assets
```

# Run

```
make run-mic-test
```

or

```
make run-wav-test
```

# Debug

Replace run with debug

```
make debug-mic-test
```

or

```
make debug-wav-test
```
