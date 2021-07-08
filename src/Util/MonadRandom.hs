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

newtype MonadRandom g a = MonadRandom {runRandom' :: g -> (a,g)}

runRandom :: g -> MonadRandom g a -> (a, g)
runRandom = flip runRandom'

instance (RandomGen g) => Functor (MonadRandom g) where
        aToB `fmap` m = MonadRandom $ \g -> let (a, g') = runRandom' m g
                                            in (aToB a, g')

instance (RandomGen g) => Applicative (MonadRandom g) where
        pure a = MonadRandom $ \g -> (a,g)
        mAtoB <*> m = MonadRandom $ \g ->
                let (aToB, g') = runRandom' mAtoB g
                    (a, g'') = runRandom' m g'
                in (aToB a,g'')

instance (RandomGen g) => Monad (MonadRandom g) where
        mA >>= aToMb = MonadRandom $ \g ->
                let (a,g') = runRandom' mA g
                in runRandom' (aToMb a) g'

getRandom :: (RandomGen g, Random a) => MonadRandom g a
getRandom = MonadRandom $ \g -> random g

getRandomR :: (RandomGen g, Random a) => (a, a) -> MonadRandom g a
getRandomR r = MonadRandom $ \g -> randomR r g

getRandoms :: (RandomGen g, Random a) => Int -> MonadRandom g [a]
getRandoms n = replicateM n getRandom

getRandomRs :: (RandomGen g, Random a) => (a, a) -> Int -> MonadRandom g [a]
getRandomRs r n = replicateM n (getRandomR r)

{-
data MonadRandom g a = NewRandom {unMonadRandom::a}
                       | WithGen {unMonadRandom::a, getGen::g}

instance (RandomGen g) => Functor (MonadRandom g) where 
        aToB `fmap` NewRandom a   = NewRandom $ aToB a
        aToB `fmap` (WithGen a g) = WithGen (aToB a) g

instance (RandomGen g) => Applicative (MonadRandom g) where
        pure = NewRandom
        NewRandom aToB <*> NewRandom a  = NewRandom $ aToB a
        WithGen aToB g <*> WithGen a g' = WithGen (aToB a) g'
        WithGen aToB g <*> NewRandom a  = WithGen (aToB a) g
        NewRandom aToB <*> WithGen a g' = WithGen (aToB a) g'

instance (RandomGen g) => Monad (MonadRandom g) where
        (NewRandom a) >>= aToM = aToM a
        (WithGen a g) >>= aToM = case aToM a of 
                                      NewRandom b -> WithGen b g
                                      WithGen b g' -> WithGen b g'
-}