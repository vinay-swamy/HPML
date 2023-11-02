## Contents

- chatbot.ipynb: The main file for training and testing the chatbot.
- hp_tuned_4k_train_iters.tar.gz - the chatbot model file trained with optimal hyperparameters.
- no_hp_4k_train_iters.tar.gz - the charbot model file trained with default hyperparameters.
- serialized_scripted_searcher.pt - the serialized scripted model
- nonPython_Bot/ - Files required for loading the model in c++.
- report.pdf - the anwser to Q1-Q3, graphs from the pytorch profiler and WandB.

### Running the c++ model

This code draws heavily from this tutorial: https://pytorch.org/tutorials/advanced/cpp_export.html

 To run the precompiled c++ model, run the following commands:

```
./nonPython_bot/build/nonPython_bot serialized_scripted_searcher.pt 
```

to recompile the model, run the following commands:

```
cd nonPython_bot
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip
rm -rf build 
mkdir build 
cd build 
cmake -DCMAKE_PREFIX_PATH=../libtorch ..
cmake --build . --config Release
```

Note that this just checks if the serialized model can be loaded correctly. To actually run the model, one would need to port tokenization preprocessing from the python to c++, which would require a lot of work.