(ns basic-nn.core
  (:require [clojure.core.matrix :as matrix :refer [dot transpose exp]]
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

(defn activation
  "Sigmoid function"
  [x] (/ 1 (+ 1 (exp (- x)))))

(defn derivative
  "Derivative of sigmoid"
  [x] (* x (- 1 x)))


(defn layer
  "The layers in our network are a curried function of weights and inputs"
  [weights]
  (fn [inputs]
    (activate (dot inputs @weights))))

(def network
  "Our network is nothing but the composition of our layers.
  Note that function composition is read right to left."
  (comp (layer synapses-1) (layer synapses-0)))

;; Thats all we need for forward propagation
;; We can now apply our network as a function of our training input
(network training-input)

;; Training is the process of gradient descent
;; For each layer we need to multiply the derivative of our layer function
;; in relation to the weights by the size of the error.

;; Because these are just functions we can run backwards as well as forwards
;; and thus derive our input from out output.

;; Just as the networks feed forward is the composition of layers,
;; the network backward propagation


(defn deltas [cost-fn]
  (reduce
   (fn [deltas layer]
     (conj deltas
           (* (derivative layer)
              (if (empty? deltas)
                (cost-fn)
                (dot (last deltas) (transpose @synapses-1))))))
   [] [(network training-input) ((layer synapses-0) training-input)]))

(defn gradient-decent [cost-fn]
  (let [deltas (deltas cost-fn)]
    (swap! synapses-1 + (dot (transpose ((layer synapses-0) training-input))
                             (first deltas)
                             ))
    (swap! synapses-0 + (dot (transpose training-input)
                             (second deltas)))
    (mean-error (errors))
    ))


(def errors (fn [] (- training-output (network training-input))))

(defn mean-error [numbers]
  (let [absolutes (map abs (flatten numbers))]
    (/ (apply + absolutes) (count absolutes))))


;; Train once
(gradient-decent errors)

;; Train 1000 passes
(dotimes [i 1000]
  (gradient-decent errors))


;; Check results
(network training-input)

;; Check results for new values
(network [0 1 0])
