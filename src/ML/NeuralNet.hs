module ML.NeuralNet
(newNeuralNet
,newNeuralNetIO
,NeuralNet
,ActivationFunction(..)
,predict
,train'
,train
) where


import Util.MonadRandom (MonadRandom, getRandomRs, runRandom)
import System.Random (RandomGen, newStdGen)
import Control.Monad (replicateM, foldM)
import Data.List (foldl')


-- | The activation functions that are used on a layer.
data ActivationFunction = Logistic
                        | ReLU
                        | Identity
                          deriving (Show,Read)
-- | A neural net
data NeuralNet a = NeuralNet {neuralNetLearningRate::Float
                             ,neuralNetInputLayer::Layer a
                             ,neuralNetHiddenLayers::[Layer a]
                             ,neuralNetOutputLayer::Layer a
                             }
                   deriving (Show,Read)
-- | A single neuron inside a layer that has inbound weights. For the input layer, these are all ones.
data Neuron a = Neuron a [Float] deriving (Show,Read)

-- | A single layer in a neural net, holding neurons and an activation function.
data Layer a = Layer {layerActivationFunction::ActivationFunction
                     ,layerNeurons::[Neuron a]
                     }
               deriving (Show,Read)

-- | Net input is the summation of component-wise multiplication of the input vectors according to the inbound weights.
newtype NetInput = NetInput Float deriving Show
-- | An activation is what the neurons provide to the next layer (or output). It is calculated by applying the activation function on the NetInput.
newtype Activation = Activation {unActivation::Float} deriving Show
-- | A delta is is calculated per neuron in the backpropagation step.
--   In the output layer, it is the application of the activation function's derivative with the net-input,
--      multiplied by the desired outcome minus the actual outcome.
--   In hidden layers, it is the aforementioned derivative value multiplied by the summation of 
--      pairwise multiplication of the neurons outbound weights and the next layer's delta.
newtype Delta = Delta {unDelta::Float} deriving Show
-- | DeltaWeight is the change of weight to add to each neuron's weights in the last step of propagating backwards.
--   It is calculated by the learning-rate times the 
--      summation of component-wise multiplication of the current layer's deltas and the previous layer's activations
newtype DeltaWeight = DeltaWeight {unDeltaWeight::Float} deriving Show


-- | Summation of component-wise multiplications of two vectors.
sumMults :: [Float] -> [Float] -> Float
sumMults (a:as) (b:bs) = a*b + as `sumMults` bs
sumMults [] [] = 0
sumMults [] bs = error $ "sumMults length mismatch. got too many bs, namely " <> show bs
sumMults as [] = error $ "sumMults product length mismatch. got too many as, namely " <> show as

-- | Apply the activation function
activate :: ActivationFunction -> Float -> Float
activate Logistic x = 1 / (1 + exp 1 ** (-x))
activate ReLU x = if x < 0 then 0 else x
activate Identity x = x

-- | Apply the derivative of an activation function to a net input
activate' :: ActivationFunction -> Float -> Float
activate' Logistic x = activate Logistic  x * (1 - activate Logistic x)
activate' ReLU x = if x < 0 then 0 else 1
activate' Identity _ = 1

{-#INLINE neuronPayload#-}
-- | Fetch whatever data was stored alongside a neuron from a previous step.
neuronPayload :: Layer a -> [a]
neuronPayload (Layer _ neurons) = map (\(Neuron p _) -> p) neurons

{-#INLINE activations#-}
-- | Fetch activations calculated for a layer.
activations :: Layer (a,Activation) -> [Activation]
activations l = map snd $ neuronPayload l

{-#INLINE deltas #-}
-- | Fetch deltas calculated for a layer.
deltas :: Layer (Delta,Activation) -> [Delta]
deltas l = map fst $ neuronPayload l

{-#INLINE layerActivations#-}
-- | Fetch a list of a layer's neuron's activations.
layerActivations :: Layer (a,Activation) -> Layer Activation
layerActivations (Layer actF neurons) = Layer actF $ map (\(Neuron (_,a) ws) -> Neuron a ws) neurons

{-#INLINE outboundWeights#-}
-- | Maps neurons to their outbound weights. 
--   Note that this should not be invoked on the output layer,
--   because their neurons have no outbound weights.
outboundWeights :: Layer a -> Layer b -> [[Float]]
outboundWeights (Layer _ curNeurons) (Layer _ nextNeurons) =
        zipWith (\ idx _y -> map (\ (Neuron _ ws) -> ws !! idx) nextNeurons) [1..(length curNeurons)] curNeurons -- start at index 1 so that bias weights are disregarded.


-- | Forward propagation of inputs.
--   Used together with `train` because it saves previous activations 
--   and net inputs which are required to calculate the deltas and derivatives.
feed :: NeuralNet () -> [[Float]] -> NeuralNet (NetInput,Activation)
feed net input = let
         Layer inActF inNeurons = neuralNetInputLayer net
         inputLayer' = Layer inActF
                        $ zipWith (\(Neuron _ ws) is ->
                                    let z = ws `sumMults` is
                                    in Neuron (NetInput z
                                              ,Activation $ activate inActF z
                                              ) ws
                                  ) inNeurons input
         hiddenLayers' = snd
                         $ foldr (\(Layer actF neurons) (lastActivations,ls)->
                                  let l = Layer actF
                                        $ map (\(Neuron _ ws) ->
                                                let z = ws `sumMults` (1:map unActivation lastActivations) -- prepend 1 for bias
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
                                let z = ws `sumMults` (1:map unActivation (activations $ last (inputLayer':hiddenLayers'))) -- prepend 1 for bias
                                in Neuron (NetInput z, Activation $ activate outActF z) ws
                              ) outNeurons
        in
        NeuralNet {neuralNetLearningRate = neuralNetLearningRate net
                  ,neuralNetInputLayer=inputLayer'
                  ,neuralNetHiddenLayers=hiddenLayers'
                  ,neuralNetOutputLayer=outputLayer'
                  }

-- | Batch train a neural net.
train' :: NeuralNet () -> [([[Float]], [Float])] -> NeuralNet ()
train' = foldl' (\net' (inputs,output) -> train net' inputs output)

-- | Trains a given neural net such that the neuron's weights change.
train :: NeuralNet () -> [[Float]] -> [Float] -> NeuralNet ()
train net inputs desiredOutputs =
        let net' = feed net inputs
            learningRate = neuralNetLearningRate net
            inputLayer = neuralNetInputLayer net'
            Layer outActF outNeurons = neuralNetOutputLayer net'
            deltaOutputLayer = Layer outActF $ zipWith (\(Neuron (NetInput netInput, actual) ws) desired -> 
                                                                  Neuron (Delta $ activate' outActF netInput * (desired - unActivation actual)
                                                                         ,actual
                                                                         ) ws
                                         )
                               outNeurons
                               desiredOutputs
            deltaHiddenLayers =
                    snd
                    $ foldr (\hiddenLayer@(Layer actF neurons) (nextLayer,ls) ->
                                let nextDeltas = deltas nextLayer
                                    outboundWs = outboundWeights hiddenLayer nextLayer
                                    hiddenLayer' = Layer actF $ zipWith (\(Neuron (NetInput netInput, a) ws) ws' ->
                                                    Neuron (Delta $ activate' actF netInput * (map unDelta nextDeltas) `sumMults` ws'
                                                           ,a
                                                           ) ws
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
                                        -- in case there are no hidden layers,
                                        --    the input layer is the previous layer to the output layer.
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

-- | Use a trained neural net to classify the input.
predict :: NeuralNet () -> [[Float]] -> [Float]
predict net inputs = map unActivation
                   $ activations
                   $ neuralNetOutputLayer
                   $ feed net inputs

-- | Apply previously found delta-weights to the neurons current weights.
--   Final step of backpropagation.
applyDeltaWeight :: Layer [DeltaWeight] -> Layer ()
applyDeltaWeight (Layer actF neurons) =
        Layer actF
              $ map (\(Neuron deltaWeights ws) ->
                        Neuron () (zipWith (+) ws (map unDeltaWeight deltaWeights))
                    ) neurons

-- | Creates a neural net with random weights.
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

-- | same as `newNeuralNet` but doesn't require you to use MonadRandom.
newNeuralNetIO :: Float
              -> Int
              -> Int
              -> [(ActivationFunction, Int)]
              -> (ActivationFunction, Int)
              -> IO (NeuralNet ())
newNeuralNetIO learningRate
               inputs
               inSize
               hiddenDesc
               outDesc = do
        g <- newStdGen
        pure $ fst
             $ runRandom g
             $ newNeuralNet learningRate inputs inSize hiddenDesc outDesc