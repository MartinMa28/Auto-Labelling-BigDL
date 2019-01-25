date
spark-submit \
--master local[4] \
--driver-memory 10g \
--class com.intel.analytics.bigdl.models.resnet.test \
/home/yilinma/Documents/IntelliJ_IDEA_Projects/Test/target/AI-Master-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
/home/yilinma/Documents/IntelliJ_IDEA_Projects/Test/core-site.xml \
/home/yilinma/Documents/IntelliJ_IDEA_Projects/Test/hbase-site.xml \
/home/yilinma/Documents/tmp/model_new_helper_API_10.obj \
kfb_512_100_test 000000001 000000051 50
date
