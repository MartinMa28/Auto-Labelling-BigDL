spark-submit \
--verbose \
--master local[2] \
--driver-memory 6g \
--class com.intel.analytics.bigdl.models.resnet.TrainKfbio \
--batchSize 2 --nEpochs 10 --learningRate 0.1 --warmupEpoch 5 \
--maxLr 3.2 --depth 50 --classes 2 \
--modelSavingPath /home/yilinma/Documents/tmp/AutoLabelingModels/juypter_model.obj
