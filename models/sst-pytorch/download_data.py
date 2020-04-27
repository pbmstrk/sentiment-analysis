import urllib.request
import re
import logging
import pandas as pd
import tensorflow_datasets as tfds

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)




def clean_text(text):
  text = text.lower()
  text = re.sub("-rrb-","-", text)
  text = re.sub("\.\.\.", "", text)
  text = re.sub("\\n", "", text)
  text = re.sub("-lrb-","-", text)
  text = re.sub("\\\\\*", "", text)
  text = re.sub("n\'t", "not", text)
  text = re.sub("\'ll", "will", text)
  text = re.sub("\'s", "is", text)
  text = re.sub("\\\\", "", text)
  text = re.sub("[\`\"]*", "", text)
  return text



if __name__ == '__main__':
 
    logger.info("Downloading data.")
    data = tfds.load('glue/sst2', batch_size=-1)

    logger.info("Processing data.")
    train_data = data['train']
    dev_data = data['validation']
    test_data = data['test']
    
    X_train = [clean_text(sentence.decode("utf-8")) for sentence in tfds.as_numpy(train_data['sentence'])]
    y_train = tfds.as_numpy(train_data['label'])

    X_dev = [clean_text(sentence.decode("utf-8")) for sentence in tfds.as_numpy(dev_data['sentence'])]
    y_dev = tfds.as_numpy(dev_data['label'])

    X_test = [clean_text(sentence.decode("utf-8")) for sentence in tfds.as_numpy(test_data['sentence'])]
    y_test = tfds.as_numpy(test_data['label'])
    
    logger.info("Saving data.")
    traindf = pd.DataFrame({"sentences": X_train, "labels": y_train})
    traindf.to_csv('./data/train.csv', index=False)

    devdf = pd.DataFrame({"sentences": X_dev, "labels": y_dev})
    devdf.to_csv('./data/dev.csv', index=False)

    testdf = pd.DataFrame({"sentences": X_test, "labels": y_test})
    testdf.to_csv('./data/test.csv', index=False)

    
    

    

