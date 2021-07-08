module ML.Perceptron
        (Neuron(..)
        ,predict
        ,train
        ,ActivationFunction(..)
        ,Weight
        ,NetInput
        ,Input
        ,Sample
        ,Bias
        ,LearningRate
        ,activate
        ,newNeuron
        ,randomNeuron
        ) where

import Util.MonadRandom
import System.Random (RandomGen)

type Weight = Float
type Input = [Float]
type Sample = (Input, Float)
type LearningRate = Float
type Bias = Float
type NetInput = Float

data ActivationFunction = Signum
                        | ReLU
                        | Sigmoid deriving Show

data Neuron = Neuron {neuronActivationFunction::ActivationFunction
                     ,neuronLearningRate::LearningRate
                     ,neoronBias::Bias
                     ,neuronWeights::[Weight]
                     }
instance Show Neuron where
        show (Neuron a lr bias ws) =
                "Neuron, Activation Function: " <> show a
                <> ", learning-rate: "          <> show lr
                <> ", bias: "                   <> show bias
                <> ", weights: "                <> show ws

activate :: ActivationFunction -> Float -> Float
activate Signum x = if x > 0 then 1 else 0
activate ReLU x = if x > 0 then x else 0
activate Sigmoid x = 1 / (1 + exp 1**(-x))

-- perform a prediction based on the activation function.
predict :: Neuron -> Input -> Float
predict p@(Neuron a _ _ _) ys = activate a (predict' p ys)

predict' :: Neuron -> Input -> NetInput
predict' (Neuron _ _ bias ws) ys = 
        if length ys /= length ws
                then error $ "mismatch - input length is " <> show (length ys) <> " but there are " <> show (length ws) <> " weights."
                else sum (zipWith (*) ws ys) + bias

-- trains by adjusting the weights to desired output.
train :: Neuron -> Sample -> Neuron
train p@(Neuron a lr bias ws) (ys, d) =
        Neuron a lr bias
        $ zipWith (\w y -> w + lr * (d - predict p ys) * y) ws ys

-- perceptron smart constructor, negating the bias.
newNeuron :: ActivationFunction -> LearningRate -> Int -> Bias -> Neuron
newNeuron a lr n bias = Neuron a lr (-bias) (replicate n 0)

randomNeuron :: RandomGen g => ActivationFunction -> LearningRate -> Int -> Bias -> MonadRandom g Neuron
randomNeuron a lr s b = do
        weights <- getRandomRs (0.0,1.0) s
        pure $ Neuron a lr (-b) weights
