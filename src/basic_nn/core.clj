(ns basic-nn.core
  (:require [clojure.core.matrix :as matrix]
            [clojure.core.matrix.random :as random]
            [clojure.core.matrix.operators :refer :all]
            [clojure.math.numeric-tower :refer [expt]]
            ))

(matrix/set-current-implementation :vectorz)

(def training-input
  [[0 0 1]
   [1 1 1]
   [1 0 1]
   [0 1 1]])

(def training-output
  (matrix/transpose
   [[0 1 1 0]]))


(defn random-array [x y]
  (matrix/array 
   (take y
         (repeatedly
          #(- (* 2 (take x (random/randoms))) 1)
          ))))

(def synaptic-weights (atom (random-array 1 3)))

(deref synaptic-weights)

(defn feed-forward 
  []
  (/ 1
     (+ 1
        (#(* % %)
            (- (matrix/dot training-input @synaptic-weights)) ))))


(defn calc-synaptic-weights [output synaptic-weights]
  (+ synaptic-weights
     (matrix/dot (matrix/transpose training-input)
                (* (- training-output output) output (- 1 output)))))


;; Train 1000 passes

(doseq [i (range 1000)]
  (reset! synaptic-weights
          (calc-synaptic-weights (feed-forward) @synaptic-weights)))


(feed-forward)

(reset! synaptic-weights
        (calc-synaptic-weights (feed-forward) @synaptic-weights))
