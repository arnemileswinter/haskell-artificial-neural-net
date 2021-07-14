module MNist.Util where 

import Data.Word (Word8)
import qualified Data.ByteString.Lazy as BL
import System.FilePath ((</>))
import Data.Binary (Get)
import Data.Binary.Get (runGet, getWord8, getInt32be, runGetIncremental, Decoder (Done, Partial, Fail))
import Control.Monad (replicateM)
import qualified Data.ByteString as BS
import Data.ByteString.Lazy.Internal (chunk, ByteString(Chunk))
import Data.Int (Int64)

fromGetInt32be :: Get Int
fromGetInt32be = fromIntegral <$> getInt32be

data LabelFileHeader =
        LabelFileHeader {labelFileHeaderMagicNumber :: Int
                        , labelFileHeaderNumberOfLabels :: Int
                        } deriving Show
getLabelFileHeader :: Get LabelFileHeader
getLabelFileHeader =
        LabelFileHeader <$> fromGetInt32be
                        <*> fromGetInt32be

data ImageFileHeader =
        ImageFileHeader {imageFileHeaderMagicNumber :: Int
                        ,imageFileHeaderNumberOfImages :: Int
                        ,imageFileHeaderRows :: Int
                        ,imageFileHeaderCols :: Int
                        } deriving Show
getImageFileHeader :: Get ImageFileHeader
getImageFileHeader =
        ImageFileHeader <$> fromGetInt32be
                        <*> fromGetInt32be
                        <*> fromGetInt32be
                        <*> fromGetInt32be

imageFileHeaderLength :: Int64
imageFileHeaderLength = 4*4
labelFileHeaderLength :: Int64
labelFileHeaderLength = 2*4

newtype Image = Image [[Word8]] deriving Show
getImage :: Int -> Int -> Get Image
getImage rows cols = replicateM rows (replicateM cols getWord8) >>= pure . Image

getImages :: Int -> Int -> BL.ByteString -> [Image]
getImages rows cols = go decoder where
        decoder = runGetIncremental (getImage rows cols)
        go :: Decoder Image -> BL.ByteString -> [Image]
        go (Done leftover _consumed img) input =
                img : go decoder (chunk leftover input)
        go (Partial cont) input =
                go (cont.takeHeadChunk $ input) (dropHeadChunk input)
        go (Fail _leftover _consumed _msg) _input = []

takeHeadChunk :: BL.ByteString -> Maybe BS.ByteString
takeHeadChunk (Chunk bs _) = Just bs
takeHeadChunk _ = Nothing

dropHeadChunk :: BL.ByteString -> BL.ByteString
dropHeadChunk (Chunk _ lbs) = lbs
dropHeadChunk _ = BL.empty
newtype Label = Label {unLabel::Int} deriving (Show)

getLabel :: Get Label
getLabel = Label . fromIntegral <$> getWord8

getLabels :: BL.ByteString -> [Label]
getLabels = go decoder where
        decoder = runGetIncremental (getLabel)
        go :: Decoder Label -> BL.ByteString -> [Label]
        go (Done leftover _consumed lbl) input = lbl : go decoder (chunk leftover input)
        go (Partial cont) input =
                go (cont.takeHeadChunk $ input) (dropHeadChunk input)
        go (Fail _leftover _consumed _msg) _input = []


toPgm:: Image -> String
toPgm (Image pxls) =
        let rows = length pxls
            cols = length (head pxls)
        in "P2\n"
           <> show rows <> " " <> show cols <> "\n"
           <> "255 \n"
           <> unlines (map (unwords.map show) pxls)

getLabeledImages :: FilePath -> String -> IO (Int,Int,[Image],[Label])
getLabeledImages dataDir prefix = do
       lblFile <- BL.readFile $ dataDir </> prefix <> "-labels-idx1-ubyte"
       imgFile <- BL.readFile $ dataDir </> prefix <> "-images-idx3-ubyte"
       -- just pattern match magic numbers to make sure they are proper.
       let (LabelFileHeader 2049 _numLabels) = runGet getLabelFileHeader lblFile
           (ImageFileHeader 2051 _numImages rows cols) = runGet getImageFileHeader imgFile
           imgs = getImages rows cols (BL.drop imageFileHeaderLength imgFile)
           lbls = getLabels (BL.drop labelFileHeaderLength lblFile)
       pure (rows,cols,imgs, lbls)

getLabeledTrainImages :: FilePath -> IO (Int,Int,[Image],[Label])
getLabeledTrainImages dataDir = getLabeledImages dataDir "train"

getLabeledTestImages :: FilePath -> IO (Int,Int,[Image],[Label])
getLabeledTestImages dataDir = getLabeledImages dataDir "train" -- "t10k"