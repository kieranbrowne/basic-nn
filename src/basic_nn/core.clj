(ns basic-nn.core
  (:require [clojure.core.matrix :as matrix]
            [clojure.core.matrix.random :as random]
            [clojure.core.matrix.operators :refer :all]
            [clojure.math.numeric-tower :refer [expt abs round]]
            ))

;; (matrix/set-current-implementation :vectorz)

(def training-data
   ;; input => output
   [ [0 0 1]   [ 0 ]
     [0 1 1]   [ 1 ]
     [1 0 1]   [ 1 ]
     [1 1 1]   [ 0 ] ])

(def training-input
  (take-nth 2 training-data))

(def training-output
  (take-nth 2 (rest training-data)))


(defn matrix-of
  "Return a matrix of results of a function"
  [function shape]
  (matrix/array
   (repeatedly (first shape)
    (if (empty? (rest shape))
      function
      #(matrix-of function (rest shape))))))


(defn random-synapse
  "Random float between -1 and 1"
  [] (dec (rand 2)))

;; synapses are mutable so I'm using atoms
(def synapses-0 (atom (matrix-of random-synapse [3 5])))
(def synapses-1 (atom (matrix-of random-synapse [5 1])))

(defn activate
  "Sigmoid function"
  [x] (/ 1 (+ 1 (matrix/exp (- x)))))

(defn deactivate
  "Derivative of sigmoid"
  [x] (* x (- 1 x)))


(defn step-forward
  "For each step in feed forward, the new layer of neurons
  are a function of the previous layer and it's synapses"
  [neurons synapses]
  (activate (matrix/dot neurons synapses)))

(defn feed-forward
  [input & synapses]
  (case (count synapses)
    0 input ; if no synapses return just the input
    1 (step-forward input @(first synapses))
    ;; otherwise recurse
    (apply
     feed-forward
     (into [(step-forward input @(first synapses))]
           (rest synapses)))))


(defn errors [training-outputs real-outputs]
  (- training-outputs real-outputs))

(defn output-deltas [targets outputs]
  (* (deactivate outputs)
     (- targets outputs)))

(defn hidden-deltas [output-deltas neurons synapses]
  (* (matrix/dot output-deltas (matrix/transpose synapses))
     (deactivate neurons)
     ))

(defn apply-deltas [synapses neurons deltas learning-rate]
  (+ synapses
     (* learning-rate
        (matrix/dot (matrix/transpose neurons) deltas))))

(defn mean-error [numbers]
  (let [absolutes (map abs (flatten numbers))]
    (/ (apply + absolutes) (count absolutes))))

(defn train
  "Train the network and return the error"
  [training-input training-output]
  (let [
        hidden-layer (step-forward training-input @synapses-0)
        output-layer (step-forward hidden-layer @synapses-1)
        output-delta (output-deltas training-output output-layer)
        hidden-delta (hidden-deltas output-delta hidden-layer @synapses-1)
        output-error (errors training-output output-layer)
        ]
    (do
      (swap! synapses-1
             #(apply-deltas % hidden-layer output-delta 1))
      (swap! synapses-0
             #(apply-deltas % training-input hidden-delta 1))
      (mean-error output-error)
      )))

(defn train-prime
  "Train the network and return the error"
  [times training-input training-output]
  (loop [times times syn0 @synapses-0 syn1 @synapses-1]
           (if (<= times 0)
             (do ; set the syn vals
               (reset! synapses-1 syn1)
               (reset! synapses-0 syn0)
               (mean-error
                (errors training-output
                        (feed-forward
                         training-input
                         synapses-0 synapses-1)))
               )
             (let [
                   hidden-layer (step-forward training-input syn0)
                   output-layer (step-forward hidden-layer syn1)
                   output-delta (output-deltas training-output output-layer)
                   hidden-delta (hidden-deltas output-delta hidden-layer syn1)
                   ]
                 (recur (dec times)
                        (apply-deltas syn0 training-input hidden-delta 1)
                        (apply-deltas syn1 hidden-layer output-delta 1)
                        ))
               )
             ))


;; Train once
(train training-input training-output)

;; Train 1000 passes
(train-prime 10000 training-input training-output)

;; Train 1000 passes
(dotimes [i 10000]
  (train training-input training-output))


;; Check results
(feed-forward training-input synapses-0 synapses-1)

;; Check results for new values
(feed-forward [0 1 0] synapses-0 synapses-1)
