module Util.MonadRandom
 (MonadRandom
 ,runRandom
 ,getRandom
 ,getRandoms
 ,getRandomR
 ,getRandomRs
 )
 where

import System.Random (RandomGen, Random (random, randomR))
import Control.Monad (replicateM)

-- | Useful to generate plenty of random values withohut carrying the generator along.
newtype MonadRandom g a = MonadRandom {unRandom :: g -> (a,g)}

-- | Leave the monadic context, yielding the random number generator and the generated output.
runRandom :: g -> MonadRandom g a -> (a, g)
runRandom = flip unRandom

instance (RandomGen g) => Functor (MonadRandom g) where
        aToB `fmap` m = MonadRandom $ \g -> let (a, g') = unRandom m g
                                            in (aToB a, g')

instance (RandomGen g) => Applicative (MonadRandom g) where
        pure a = MonadRandom $ \g -> (a,g)
        mAtoB <*> m = MonadRandom $ \g ->
                let (aToB, g') = unRandom mAtoB g
                    (a, g'') = unRandom m g'
                in (aToB a,g'')

instance (RandomGen g) => Monad (MonadRandom g) where
        mA >>= aToMb = MonadRandom $ \g ->
                let (a,g') = unRandom mA g
                in unRandom (aToMb a) g'

-- | Retrieve a single random
getRandom :: (RandomGen g, Random a) => MonadRandom g a
getRandom = MonadRandom $ \g -> random g

-- | Generate a single random within a specific range
getRandomR :: (RandomGen g, Random a) => 
              (a, a) -- ^ The lower bound and upper bound of the randoms to generate.
              -> MonadRandom g a
getRandomR r = MonadRandom $ \g -> randomR r g

-- | Generate a list of randoms.
getRandoms :: (RandomGen g, Random a) => Int -> MonadRandom g [a]
getRandoms n = replicateM n getRandom

-- | Generate a list of randoms
getRandomRs :: (RandomGen g, Random a) => 
               (a, a) -- ^ The lower bound and upper bound of the randoms to generate.
               -> Int -- ^ The amount of randoms to generate.
               -> MonadRandom g [a]
getRandomRs r n = replicateM n (getRandomR r)