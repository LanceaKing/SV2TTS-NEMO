@echo off

type data\train_clean_100_speaker.json data\train_clean_360_speaker.json data\LJSpeech-1.1\ljspeech_train_speaker.json > data\train_speaker.json
type data\train_clean_100_text.json data\train_clean_360_text.json data\LJSpeech-1.1\ljspeech_train_text.json > data\train_text.json

type data\dev_clean_speaker.json data\dev_clean_speaker.json data\LJSpeech-1.1\ljspeech_val_speaker.json > data\dev_speaker.json
type data\dev_clean_text.json data\dev_clean_text.json data\LJSpeech-1.1\ljspeech_val_text.json > data\dev_text.json
