(ns basic-nn.core
  (:require [clojure.core.matrix :as matrix]
            [clojure.core.matrix.random :as random]
            [clojure.core.matrix.operators :refer :all]
            [clojure.math.numeric-tower :refer [expt abs]]
            ))

(matrix/set-current-implementation :vectorz)

(def training-input
  [[0 0 1]
   [0 1 1]
   [1 0 1]
   [1 1 1]])

(def training-output
  (matrix/transpose
   [[0 1 1 0]]))


(defn random-synapses
  "Create a matrix of random synaptic weights."
  [from to]
  (matrix/array
   (take from
         (repeatedly
          #(- (* 2 (take to (random/randoms))) 1)
          ))))

;; synapses are mutable so I'm using atoms
(def synapses-0 (atom (random-synapses 3 4)))
(def synapses-1 (atom (random-synapses 4 1)))

;; sigmoid fn
(defn sigmoid [x]
  (/ 1
     (+ 1 (matrix/exp (- x)))))

;; derivative of sigmoid
(defn deriv [x]
  (* x (- 1 x)))


(defn step-forward
  "For each step in feed forward, the new layer of neurons
  are a function of the previous layer and it's synapses"
  [neurons synapses]
  (sigmoid (matrix/dot neurons synapses)))


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
  (* (deriv outputs)
     (- targets outputs)))

(defn hidden-deltas [output-deltas neurons synapses]
  (* (matrix/dot output-deltas (matrix/transpose synapses))
     (deriv neurons)
     ))

(defn apply-deltas [synapses neurons deltas learning-rate]
  (+ @synapses
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
      (reset! synapses-1
              (apply-deltas synapses-1 hidden-layer output-delta 1))
      (reset! synapses-0
              (apply-deltas synapses-0 training-input hidden-delta 1))
      (mean-error output-error)
      )))

;; Train once
(train training-input training-output)

;; Train 1000 passes
(doseq [i (range 10000)]
  (train training-input training-output))

;; Check results
(feed-forward training-input synapses-0 synapses-1)

;; Check results
(feed-forward [[1 0 0]] synapses-0 synapses-1)
