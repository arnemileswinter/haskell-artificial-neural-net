module Main where

import System.Environment (getArgs, withArgs)
import MNist.Util (Image(..),Label(..),getLabeledTrainImages,getLabeledTestImages)
import System.Random (newStdGen)
import Util.MonadRandom (runRandom)
import ML.NeuralNet (newNeuralNet, ActivationFunction (..), train', predict, NeuralNet)
import Control.Monad (foldM_)

type NormalFlatImage = [Float]

flattenAndNormalizeImage :: Image -> NormalFlatImage
flattenAndNormalizeImage (Image rows)= concatMap (map (\col -> fromIntegral col/255)) rows

labelToOutput :: Label -> [Float]
labelToOutput (Label num) = map (\d -> if num == d then 1.0 else 0.0) [0..9]

train :: FilePath -> IO (NeuralNet ())
train dirPath = do
        (rows,cols, trainImgs, trainLbls) <- getLabeledTrainImages dirPath
        g <- newStdGen
        let pixelCount = rows * cols
            (n, _g') = runRandom g $ newNeuralNet 0.05   -- learning rate
                                        pixelCount       -- there is one input layer neuron for each pixel
                                        [(ReLU, 16),(Identity,14)] -- hidden layer is one sigmoid of 15 neurons
                                        (Logistic, 10)   -- output layer is 10 digits.
            trainFloatImgs = map flattenAndNormalizeImage trainImgs
            trainLabelsToOutput = map labelToOutput trainLbls
            n' = train' n $ take 2 $ zip trainFloatImgs trainLabelsToOutput
        pure n'

bestPrediction :: [(Float,Int)] -> Int
bestPrediction = snd . foldr (\(prevBest,prevIdx) (prediction,idx) ->
                                 if prediction > prevBest
                                         then (prediction,idx)
                                         else (prevBest,prevIdx)
                             ) (0,0)

test :: FilePath -> NeuralNet () -> IO ()
test dirPath net = do
        (_rows,_cols,testImgs,testLbls) <- getLabeledTestImages dirPath
        let testFloatImgs = map flattenAndNormalizeImage testImgs
        foldM_ (\(fails,iterations) (image,Label label) -> do
                let prediction = predict net image
                    bestPred = bestPrediction $ zip prediction [0..]
                print $ show prediction <>" " <> show bestPred <> " and expected " <> show (labelToOutput $ Label label)
                pure (zipWith (+) fails (labelToOutput $ Label label)
                     ,iterations + 1
                     )
               )
               (map (const 0) [0..9 :: Float], 0 :: Int)
               $ take 1000 $ zip testFloatImgs testLbls

help :: IO ()
help = putStrLn $ "usage:\n"
                  <> "\t./mnist train <path-to-data-dir> <path-to-save-net>\n"
                  <> "\t./mnist test <path-to-data-dir> <path-to-saved-net>"

main :: IO ()
main = do
        args <- getArgs
        if length args /= 3
           then help
           else do
                let [_todo,dirPath,_netPath] = args
                net <- train dirPath
                test dirPath net
                -- in case todo of 
                --         "train" -> do net <- train dirPath
                --                       writeFile netPath (show net)
                --         "test" -> do 
                --                 net <- read <$> readFile netPath
                --                 test dirPath net
                --         _ -> help
