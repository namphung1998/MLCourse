const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {
  constructor(features, labels, options) {
    // features and labels are tf.Tensor's
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.mseHistory = [];

    //override first param with fields from second param
    this.options = Object.assign({ 
      learningRate: 0.1, 
      iterations: 1000,
    }, options);

    this.weights = tf.zeros([this.features.shape[1], 1]);
  }

  train() {
    const batchNum = Math.floor(this.features.shape[0] / this.options.batchSize);

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchNum; j++) {
        const startIndex = j * this.options.batchSize;
        const { batchSize } = this.options;

        const featureSlice = this.features.slice([startIndex, 0], [batchSize, -1]);
        const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);

        this.gradientDescent(featureSlice, labelSlice);
      }

      this.recordMSE();
      this.updateLearningRate();
    }
  }

  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights);
    const differences = currentGuesses.sub(labels);

    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  test(testFeatures, testLabels) {
    // testFeatures and testLabels are tf.Tensor's
    testFeatures = this.processFeatures(testFeatures);
    testLabels = tf.tensor(testLabels);

    const predictions = testFeatures.matMul(this.weights);

    const res = testLabels.sub(predictions) 
      .pow(2)
      .sum()
      .get();

    const tot = testLabels.sub(testLabels.mean())
      .pow(2)
      .sum()
      .get();

    return 1 - res / tot;
  }

  processFeatures(features) {
    features = tf.tensor(features);

    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      features = this.standardize(features);
    }

    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return features;
  }

  standardize(features) {
    const { mean, variance } = tf.moments(features, 0);

    this.mean = mean;
    this.variance = variance;

    return features.sub(mean).div(variance.pow(0.5));
  }

  recordMSE() {
    const mse = this.features.matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .get();

    this.mseHistory.unshift(mse);
  }

  updateLearningRate() {
    if (this.mseHistory.length < 2) return;

    if (this.mseHistory[0] > this.mseHistory[1]) { //just went up
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }

  predict(observations) {
    // observations is an array of arrays 
    // each row: [horsepower, weight, displacement]

    return this.processFeatures(observations).matMul(this.weights);
  }
}

module.exports = LinearRegression;