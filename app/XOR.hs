module Main where

import System.Random (newStdGen)
import Util.MonadRandom (runRandom)
import ML.NeuralNet (newNeuralNet, ActivationFunction(..),predict,train')

main :: IO ()
main = do
     g <- newStdGen
     -- get a NeuralNet
     let (n, _) = runRandom g $ newNeuralNet (0.3)         -- Learning rate is 0.3
                                              1             -- There is one incoming edge for each input neuron. 
                                              2             -- There are 2 input neurons
                                              [(ReLU, 5)]   -- There is a single hidden layer with the ReLU activation and 5 neurons
                                              (Logistic, 1) -- The output layer is a single sigmoid-activated neuron

     -- train it with a million samples that describe the XOR function.
     let nt = train' n $ take (10^(6::Int)) $ cycle [([[1],[1]],[0])
                                              ,([[1],[0]],[1])
                                              ,([[0],[1]],[1])
                                              ,([[0],[0]],[0])
                                              ]
     -- print the outcomes.
     print $ predict nt [[1],[1]]
     print $ predict nt [[0],[1]]
     print $ predict nt [[1],[0]]
     print $ predict nt [[0],[0]]
