module XOR where

import System.Random (newStdGen)
import Util.MonadRandom (runRandom)
import ML.NeuralNet (newNeuralNet, ActivationFunction(..),predict,train')

main :: IO ()
main = do
     g <- newStdGen
     let (n, _') = runRandom g $ newNeuralNet (0.3) 1 2 [(ReLU, 3)] (Sigmoid, 1)
     let nt = train' n $ take 1000000 $ cycle $ [([[1],[1]],[0])
                       ,([[1],[0]],[1])
                       ,([[0],[1]],[1])
                       ,([[0],[0]],[0])
                       ]
     print $ predict nt [[1],[1]]
     print $ predict nt [[0],[1]]
     print $ predict nt [[1],[0]]
     print $ predict nt [[0],[0]]
