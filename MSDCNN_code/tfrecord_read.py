import tensorflow as tf

def read_and_decode(filename):
    # 创建一个reader来读取TFRecord文件中的样例
    reader = tf.TFRecordReader()
    # 创建一个队列来维护输入文件列表
    filename_queue = tf.train.string_input_producer([filename])
    # 从文件中读出一个样例，也可以使用read_up_to函数一次性读取多个样例
    _, serialized_example = reader.read(filename_queue)
    # 解析读入的一个样例，如果需要解析多个，可以用parse_example函数
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    # 将字符串解析成图像对应的像素数组
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [28, 28, 1])
    # 将标签转化为整数类型
    label = tf.cast(features['label'], tf.int32)
    return image, label