spark-submit \
--master "spark://yilinma-ubuntu.sh.intel.com:7077" \
--driver-memory 8g \
--class com.intel.analytics.bigdl.models.resnet.ComparisonTest \
--conf 'spark.driver.extraJavaOptions=-Dbigdl.engineType=mkldnn' \
--conf 'spark.executor.extraJavaOptions=-Dbigdl.engineType=mkldnn' \
--executor-cores 12 \
--total-executor-cores 24 \
/home/yilinma/Documents/IntelliJ_IDEA_Projects/Test/target/AI-Master-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
/home/yilinma/Documents/tmp/model_new_helper_API_10.obj \
