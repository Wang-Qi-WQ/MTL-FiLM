Implementation of The MT-FiLM Piano Transcription


Tutorials

All of the implementation parameters are modified in the "constant_macro.py" script, which includes paths, training and dataset configs. Please refer the parameter details in the "" script.

For clarity, the address directory in this tutorial is as the following. You might need to specify the path of "./home_address/dataset", "./home_address/model" and "./home_address/eva", which are the "DATASET_PATH", "MODEL_PATH" and "EVA_LOG_PATH" in the "constant_macro.py".
    ./home_address
        /dataset
            /maestro-v3.0.0
                /(MAESTRO raw files, include a ".json" ，a ".csv" files, and ".wav" and ".midi" files in each year's folder.)
            /cache
                /cache001
                    /(Preprocessed ".pth" files, include the input CQT spectrogram and the ground truth piano rolls.)
        /model
            /ckp
                /ckp001.pt
                /ckp002.pt
                ...
            /log
                /log001
                    /(Metric and loss values during the training and validation on every epoch.)
                    ...
                /log002
                ...
        /eva
            /model001
                /metrics.txt (These metrics are evaluated on every complete music pieces. Then calculate the mean value as the final result.)
                /evaluation_piano_rolls.pt
                ...
        /codes
            /(sources codes)

This project is based on the MAESTRO V3.0.0 dataset. You might need to download the full dataset on your own. The MAESTRO official download link is:
https://magenta.tensorflow.org/datasets/maestro#download


Evaluate From Checkpoint

A model checkpoint is provided as ".pth" in this project. Before the evaluation, you might need to put the model checkpoint in the path of the "./home_address/model/ckp/".
During the evaluation, the shape of the input spectrogram is [eva_batch_size, 3200, 352]. We also use the "half_stride" strategy in the transcription process. Please refer the "piano_transcription()" function in the "transcribe.py" for further detail informs.

To evaluate the model, please run the "transcribe.py". The output and the ground truth piano rolls will be saved as "./home_address/eva/model001/evaluation_piano_rolls.pt". The evaluation metrics will be saved as "./home_address/eva/model001/metrics.txt".

Train From Scratch
Most of the default training parameters are already prepared in the "constant_macro.py". You only need to specify all the dataset and model paths in the "constant_macro.py".
During the training and validation, the shape of the input spectrogram is [train_batch_size(or validate_batch_size), 400, 352]. The evaluation metrics will be saved in "./home_address/model/log/".

The training process will generate local cache ".pth" files before the first running. The generated caches will take at least 200GB local storage after the dataset preprocessing.


Requirements
pretty_midi, mir_eval, librosa
