require('@tensorflow/tfjs-node');
// const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const plot = require('node-remote-plot');

const LinearRegression = require('./linear-regression');

let { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'displacement', 'weight'],
  labelColumns: ['mpg']
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 3,
  batchSize: 10
});

regression.train();

// console.log('Updated M is', regression.weights.get(1, 0), 'updated B is', regression.weights.get(0, 0));

const r2 = regression.test(testFeatures, testLabels);

plot({
  x: regression.mseHistory.reverse(),
  xLabel: 'Iteration #',
  yLabel: 'Mean squared error'
});

console.log('R2 is', r2);

regression.predict([
  [120, 380, 2]
]).print();
