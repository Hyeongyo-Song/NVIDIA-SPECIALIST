import tensorflow as tf

saved_model_dir = 'C:/yolov7-main/yolov7-main/best_tf/'

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open('C:/yolov7-main/yolov7-main/new/converted_model.tflite', 'wb').write(tflite_model)
