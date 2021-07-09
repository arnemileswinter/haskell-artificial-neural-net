module ML.NeuralNet
(newNeuralNet
,NeuralNet
,makeNeuralNet
,ActivationFunction(..)
,predict
,train'
,train
) where


import Util.MonadRandom (MonadRandom, getRandomRs, runRandom)
import System.Random (RandomGen, newStdGen)
import Control.Monad (replicateM, foldM)
import Data.List (foldl')

newtype Activation = Activation {unActivation::Float} deriving Show
newtype NetInput = NetInput Float deriving Show
newtype Delta = Delta {unDelta::Float} deriving Show
newtype DeltaWeight = DeltaWeight {unDeltaWeight::Float} deriving Show

data Neuron a = Neuron a [Float] deriving (Show,Read)
data ActivationFunction = Sigmoid
                        | ReLU
                        | Identity
                   deriving (Show,Read)
data Layer a = Layer {layerActivationFunction::ActivationFunction
                   ,layerNeurons::[Neuron a]
                   }
                   deriving (Show,Read)
data NeuralNet a = NeuralNet {neuralNetLearningRate::Float
                           ,neuralNetInputLayer::Layer a
                           ,neuralNetHiddenLayers::[Layer a]
                           ,neuralNetOutputLayer::Layer a
                           }
                   deriving (Show,Read)
vmult :: [Float] -> [Float] -> Float
vmult (a:as) (b:bs) = a*b + vmult as bs
vmult [] [] = 0
vmult [] bs = error $ "vmult length mismatch. got too many bs, namely " <> show bs
vmult as [] = error $ "vmult product length mismatch. got too many as, namely " <> show as

activate :: ActivationFunction -> Float -> Float
activate Sigmoid x = 1 / (1 + exp 1 ** (-x))
activate ReLU x = if x < 0 then 0 else x
activate Identity x = x

activate' :: ActivationFunction -> Float -> Float
activate' Sigmoid x = (1 / (1 + exp 1 ** (-x))) * (1 - 1 / (1 + exp 1 ** (-x)))
activate' ReLU x = if x < 0 then 0 else 1
activate' Identity _ = 1

activations :: Layer (a,Activation) -> [Activation]
activations (Layer _ neurons) = map (\(Neuron (_,a) _) -> a) neurons

layerActivations :: Layer (a,Activation) -> Layer Activation
layerActivations (Layer actF neurons) = Layer actF $ map (\(Neuron (_,a) ws) -> Neuron a ws) neurons

deltas :: Layer (Delta,Activation) -> [Delta]
deltas (Layer _ neurons) = map (\(Neuron (d,_) _) -> d) neurons

neuronPayload :: Layer a -> [a]
neuronPayload (Layer _ neurons) = map (\(Neuron p _) -> p) neurons

outboundWeights :: Layer a -> Layer b -> [[Float]]
outboundWeights (Layer _ curNeurons) (Layer _ nextNeurons) =
        zipWith (\ idx _y -> map (\ (Neuron _ ws) -> ws !! idx) nextNeurons) [1..(length curNeurons)] curNeurons -- start at index 1 so that bias weights are disregarded.

feed :: NeuralNet () -> [[Float]] -> NeuralNet (NetInput,Activation)
feed net input = let
         Layer inActF inNeurons = neuralNetInputLayer net
         inputLayer' = Layer inActF
                        $ zipWith (\(Neuron _ ws) is ->
                                    let z = ws `vmult` is
                                    in Neuron (NetInput z
                                              ,Activation $ activate inActF z
                                              ) ws
                                  ) inNeurons input
         hiddenLayers' = snd
                         $ foldr (\(Layer actF neurons) (lastActivations,ls)->
                                  let l = Layer actF
                                        $ map (\(Neuron _ ws) ->
                                                let z = ws `vmult` (1:map unActivation lastActivations) -- prepend 1 for bias
                                                in Neuron (NetInput z
                                                          ,Activation $ activate actF z
                                                          ) ws
                                              ) neurons
                                  in (activations l, ls++[l])
                                  )
                                  (activations inputLayer',[])
                                  (reverse $ neuralNetHiddenLayers net)
         Layer outActF outNeurons = neuralNetOutputLayer net
         outputLayer' = Layer outActF
                        $ map (\(Neuron _ ws) ->
                                let z = ws `vmult` (1:map unActivation (activations $ last (inputLayer':hiddenLayers'))) -- prepend 1 for bias
                                in Neuron (NetInput z, Activation $ activate outActF z) ws
                              ) outNeurons
        in
        NeuralNet {neuralNetLearningRate = neuralNetLearningRate net
                  ,neuralNetInputLayer=inputLayer'
                  ,neuralNetHiddenLayers=hiddenLayers'
                  ,neuralNetOutputLayer=outputLayer'
                  }

train' :: NeuralNet () -> [([[Float]], [Float])] -> NeuralNet ()
train' = foldl' (\net' (inputs,output) -> train net' inputs output)

train :: NeuralNet () -> [[Float]] -> [Float] -> NeuralNet ()
train net inputs desiredOutputs =
        let net' = feed net inputs
            learningRate = neuralNetLearningRate net
            inputLayer = neuralNetInputLayer net'
            Layer outActF outNeurons = neuralNetOutputLayer net'
            deltaOutputLayer = Layer outActF $ zipWith (\(Neuron (NetInput netInput, actual) ws) desired ->
                                          Neuron (Delta $ activate' outActF netInput * (desired - unActivation actual), actual) ws
                                         )
                               outNeurons
                               desiredOutputs
            deltaHiddenLayers =
                    snd
                    $ foldr (\hiddenLayer@(Layer actF neurons) (nextLayer,ls) ->
                                let nextDeltas = deltas nextLayer
                                    outboundWs = outboundWeights hiddenLayer nextLayer
                                    hiddenLayer' = Layer actF $ zipWith (\(Neuron (NetInput netInput, a) ws) ws' ->
                                        Neuron (Delta $ activate' actF netInput * (map unDelta nextDeltas) `vmult` ws', a) ws
                                        ) neurons outboundWs
                                in (hiddenLayer',hiddenLayer':ls)
                             )
                             (deltaOutputLayer,[])
                             (neuralNetHiddenLayers net')
            outputDeltaWeights =
                    Layer outActF
                          $ map (\(Neuron (Delta d,_) ws) ->
                                Neuron (map (DeltaWeight . (*d))
                                        (1 -- one for bias
                                        :if null deltaHiddenLayers
                                            then map unActivation $ activations inputLayer
                                            else map unActivation $ activations $ last deltaHiddenLayers
                                        )) ws
                                )
                                (layerNeurons deltaOutputLayer)
            hiddenDeltaWeights =
                    snd
                    $ foldr (\(Layer actF neurons) (prevLayers,ls) ->
                                (init prevLayers
                                ,Layer actF (map (\(Neuron (Delta d,_) ws) ->
                                        Neuron (map (DeltaWeight . (*d))
                                                (1 -- one for bias
                                                :map unActivation (neuronPayload $ last prevLayers))
                                                ) ws
                                        ) neurons):ls)
                             )
                             (if null deltaHiddenLayers
                                then [layerActivations inputLayer]
                                else layerActivations inputLayer:map layerActivations (init deltaHiddenLayers)
                             ,[]
                             ) deltaHiddenLayers
            outputLayer' = applyDeltaWeight outputDeltaWeights
            hiddenLayers' = map applyDeltaWeight hiddenDeltaWeights
        in NeuralNet{neuralNetLearningRate=learningRate
                    ,neuralNetInputLayer=neuralNetInputLayer net
                    ,neuralNetHiddenLayers=hiddenLayers'
                    ,neuralNetOutputLayer=outputLayer'
                    }

predict :: NeuralNet () -> [[Float]] -> [Float]
predict net inputs = map unActivation
                   $ activations
                   $ neuralNetOutputLayer
                   $ feed net inputs

applyDeltaWeight :: Layer [DeltaWeight] -> Layer ()
applyDeltaWeight (Layer actF neurons) =
        Layer actF
              $ map (\(Neuron deltaWeights ws) ->
                        Neuron () (zipWith (+) ws (map unDeltaWeight deltaWeights))
                    ) neurons

newNeuralNet :: RandomGen g =>
                Float                          -- ^ The network's learning rate.
                -> Int                         -- ^ number of inputs for each input layer neuron.
                -> Int                         -- ^ number of input layer neurons.
                -> [(ActivationFunction,Int)]  -- ^ descriptions of hidden layers. 
                                               --   Pairings of activation functions and number of neurons.
                -> (ActivationFunction,Int)    -- ^ Activation function and number of neurons of the output layer.
                -> MonadRandom g (NeuralNet ())
newNeuralNet learningRate
             inputs
             inSize
             hiddenDesc
             (outAct,outSize) = do
        let inputNeurons = replicate inSize $ Neuron () $ replicate inputs 1
        let inputLayer = inputNeurons
        outputNeurons <- replicateM outSize $ Neuron () <$> getRandomRs (0.0,1.0) (last (inSize:map snd hiddenDesc) + 1) -- one added for bias weight.
        hiddenlayers <- reverse.snd <$> foldM (\(prevSize,ls) (actF,curSize) -> do
                                neurons <- replicateM curSize $ Neuron () <$> getRandomRs (0.0,1.0) (prevSize + 1) -- one added for bias weight.
                                pure (curSize, Layer actF neurons:ls)
                ) (inSize,[]) hiddenDesc
        return $ NeuralNet {neuralNetLearningRate=learningRate
                           ,neuralNetInputLayer=Layer Identity inputLayer
                           ,neuralNetOutputLayer=Layer outAct outputNeurons
                           ,neuralNetHiddenLayers=hiddenlayers
                           }

makeNeuralNet :: Float
              -> Int
              -> Int
              -> [(ActivationFunction, Int)]
              -> (ActivationFunction, Int)
              -> IO (NeuralNet ())
makeNeuralNet learningRate
             inputs
             inSize
             hiddenDesc
             outDesc = do
        g <- newStdGen
        pure $ fst
             $ runRandom g
             $ newNeuralNet learningRate inputs inSize hiddenDesc outDesc