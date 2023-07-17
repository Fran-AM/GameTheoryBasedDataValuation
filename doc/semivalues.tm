<TeXmacs|2.1.2>

<style|generic>

<\body>
  <subsection|Mixing sampling strategies for semi-values>

  We begin by rewriting the combinatorial definition of Shapley value:

  <\eqnarray*>
    <tformat|<table|<row|<cell|v<rsub|sh><around*|(|i|)>>|<cell|=>|<cell|<frac|1|N>*<big|sum><rsub|S\<subseteq\>D<rsub|-i>><choose|N-1|<around*|\||S|\|>><rsup|-1>*<around*|[|U<around*|(|S<rsub|+i>|)>-U<around*|(|S|)>|]>,>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|S\<subseteq\>D<rsub|-i>>w<rsub|sh><around*|(|<around*|\||S|\|>|)>*\<delta\><rsub|i><around*|(|S|)>,>>>>
  </eqnarray*>

  where

  <\equation*>
    \<delta\><rsub|i><around*|(|S|)>\<assign\>U<around*|(|S<rsub|+i>|)>-U<around*|(|S|)>.
  </equation*>

  The natural Monte Carlo approximation is then to sample
  <math|S<rsub|j>\<sim\>\<cal-U\><around*|(|D<rsub|-i>|)>> and let

  <\eqnarray*>
    <tformat|<table|<row|<cell|<wide|v|^><rsub|sh,unif><around*|(|i|)>>|<cell|=>|<cell|<frac|2<rsup|N-1>|m>*<big|sum><rsub|j=1><rsup|m>w<rsub|sh><around*|(|<around*|\||S<rsub|j>|\|>|)>*\<delta\><rsub|i><around*|(|S<rsub|j>|)>>>|<row|<cell|>|<cell|\<longrightarrow\>>|<cell|<below|\<bbb-E\>|S\<sim\>\<cal-U\><around*|(|D<rsub|-i>|)>><around*|[|w<rsub|sh><around*|(|<around*|\||S|\|>|)>*\<delta\><rsub|i><around*|(|S|)>|]>*2<rsup|N-1>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|S\<subseteq\>D<rsub|-i>>w<rsub|sh><around*|(|<around*|\||S|\|>|)>*\<delta\><rsub|i><around*|(|S|)>>>>>
  </eqnarray*>

  So, because of the sampling strategy we chose, we had to add a coefficient
  <math|2<rsup|N-1>> in order to recover the value of <math|v<rsub|i>> in the
  limit. We can call this coefficient <math|w<rsub|unif>\<assign\>2<rsup|N-1>>.
  At every step of the MC algorithm we:

  <\algorithm>
    sample <math|S<rsub|j>>

    compute the marginal

    compute the product of coefficients for the sampler and the method
    <math|w<rsub|unif>*w<rsub|sh><around*|(|<around*|\||S<rsub|j>|\|>|)>>

    update the running average for <math|<wide|v|^><rsub|unif,sh>>
  </algorithm>

  Consider now the alternative choice of uniformly sampling permutations
  <math|\<sigma\><rsub|j>\<sim\>\<cal-U\><around*|(|\<Pi\><around*|(|D|)>|)>>.
  Recall that we let <math|S<rsub|i><rsup|\<sigma\>>> be the set of indices
  preceding <math|i> in the permutation <math|\<sigma\>>. Then

  <\eqnarray*>
    <tformat|<table|<row|<cell|<wide|v|^><rsub|sh,per><around*|(|i|)>>|<cell|=>|<cell|<frac|1|m>*<big|sum><rsub|j=1><rsup|m>w<rsub|per><around*|(|<around*|\||S<rsub|i><rsup|\<sigma\><rsub|j>>|\|>|)>*\<delta\><rsub|i><around*|(|S<rsub|i><rsup|\<sigma\><rsub|j>>|)>>>|<row|<cell|>|<cell|\<longrightarrow\>>|<cell|<below|\<bbb-E\>|\<sigma\>\<sim\>\<cal-U\><around*|(|\<Pi\><around*|(|D|)>|)>><around*|[|w<rsub|per><around*|(|<around*|\||S<rsub|i><rsup|\<sigma\>>|\|>|)>*\<delta\><rsub|i><around*|(|S<rsub|i><rsup|\<sigma\>>|)>|]>>>|<row|<cell|>|<cell|=>|<cell|<frac|1|N!>*<big|sum><rsub|\<sigma\>\<in\>\<Pi\><around*|(|D<rsub|-i>|)>>w<rsub|per><around*|(|<around*|\||S<rsub|i><rsup|\<sigma\>>|\|>|)>*<around*|[|U<around*|(|S<rsub|i><rsup|\<sigma\>>\<cup\><around*|{|i|}>|)>-U<around*|(|S<rsub|i><rsup|\<sigma\>>|)>|]>>>|<row|<cell|>|<cell|<above|=|<around*|(|\<star\>|)>>>|<cell|<frac|1|N!>*<big|sum><rsub|S\<subset\>D<rsub|-i>>w<rsub|per><around*|(|<around*|\||S|\|>|)>*<around*|(|N-1-<around*|\||S|\|>|)>!*<around*|\||S|\|>!*<around*|[|U<around*|(|S<rsub|+i>|)>-U<around*|(|S|)>|]>>>|<row|<cell|>|<cell|=>|<cell|<frac|1|N>*<big|sum><rsub|S\<subset\>D<rsub|-i>>w<rsub|per><around*|(|<around*|\||S|\|>|)>*<choose|N-1|<around*|\||S|\|>><rsup|-1>*\<delta\><rsub|i><around*|(|S|)>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|S\<subset\>D<rsub|-i>>w<rsub|per><around*|(|<around*|\||S|\|>|)>*w<rsub|sh><around*|(|<around*|\||S|\|>|)>*\<delta\><rsub|i><around*|(|S|)>>>|<row|<cell|>|<cell|=>|<cell|v<rsub|sh><around*|(|i|)><text|,
    if >w<rsub|per>\<equiv\>1,>>>>
  </eqnarray*>

  So we see that for permutation sampling we can choose a constant
  coefficient of 1, and simply compute an average of marginals, and we will
  recover <math|v<rsub|sh>>. At step <math|<around*|(|\<star\>|)>> we have
  counted the number of permutations before a fixed position of index
  <math|i> and after it, because the utility does not depend on ordering.
  This allows us to sum over sets instead of permutations.

  However, for consistency with the previous algorithm, we will choose
  <math|w<rsub|per><around*|(|k|)>=w<rsub|sh><around*|(|k|)><rsup|-1>> and
  include a term <math|w<rsub|sh>> in the averages, to cancel it
  out:<\footnote>
    From a numerical point of view, this is a bad idea, since it means
    computing very large numbers and then dividing one by the other. However,
    it makes the implementation very homogeneous.
  </footnote>

  <\algorithm>
    sample <math|\<sigma\><rsub|j>>

    compute the marginal <math|\<delta\><rsub|i><around*|(|S<rsub|i><rsup|\<sigma\><rsub|j>>|)>>

    compute the product of coefficients for the sampler and the method
    <math|w<rsub|per><around*|(|<around*|\||S<rsub|i><rsup|\<sigma\>>|\|>|)>*w<rsub|sh><around*|(|<around*|\||S<rsub|i><rsup|\<sigma\>>|\|>|)>=1>

    update the running average for <math|<wide|v|^><rsub|per,sh>>
  </algorithm>

  Let's look now at general semi-values, which are of the form:

  <\eqnarray*>
    <tformat|<table|<row|<cell|v<rsub|semi><around*|(|i|)>>|<cell|=>|<cell|<big|sum><rsub|k=0><rsup|n-1><wide|w|~><around*|(|k|)>*<big|sum><rsub|S\<subseteq\>D<rsub|-i><rsup|<around*|(|k|)>>>\<delta\><rsub|i><around*|(|S|)>,>>>>
  </eqnarray*>

  where <math|D<rsup|<around*|(|k|)>><rsub|-i>\<assign\><around*|{|S\<subseteq\>D<rsub|-i>:<around*|\||S|\|>=k|}>>
  and <math|<big|sum><wide|w|~><around*|(|k|)>=1>. Note that taking
  <math|<wide|w|~><around*|(|k|)>=w<rsub|sh>> we arrive at <math|v<rsub|sh>>
  simply by splitting the sum in <math|v<rsub|sh>> by size of subsets
  <math|S\<subset\>D<rsub|-i>>. Another choice is
  <math|w<rsub|bzf>\<assign\>2<rsup|-<around*|(|N-1|)>>>.

  Let's sample permutations <math|\<sigma\><rsub|j>\<sim\>\<cal-U\><around*|(|\<Pi\><around*|(|D|)>|)>>
  and approxiamte <math|v<rsub|bzf>>:

  <\eqnarray*>
    <tformat|<table|<row|<cell|<wide|v|^><rsub|bzf,per><around*|(|i|)>>|<cell|=>|<cell|<frac|1|m>*<big|sum><rsub|j=1><rsup|m>w<rsub|bzf>*w<rsub|per><around*|(|<around*|\||S<rsub|i><rsup|\<sigma\><rsub|j>>|\|>|)>*\<delta\><rsub|i><around*|(|S<rsub|i><rsup|\<sigma\><rsub|j>>|)>>>|<row|<cell|>|<cell|\<longrightarrow\>>|<cell|<below|\<bbb-E\>|\<sigma\>\<sim\>\<cal-U\><around*|(|\<Pi\><around*|(|D|)>|)>><around*|[|w<rsub|bzf>*w<rsub|per><around*|(|<around*|\||S<rsub|i><rsup|\<sigma\>>|\|>|)>*\<delta\><rsub|i><around*|(|S<rsub|i><rsup|\<sigma\>>|)>|]>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|S\<subset\>D<rsub|-i>>w<rsub|bzf>*w<rsub|per><around*|(|<around*|\||S|\|>|)>*<frac|1|N>*<choose|N-1|<around*|\||S|\|>><rsup|-1>*\<delta\><rsub|i><around*|(|S|)>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|k=0><rsup|n-1>w<rsub|bzf>*w<rsub|per><around*|(|k|)>*w<rsub|sh><around*|(|k|)>*<big|sum><rsub|S\<subseteq\>D<rsub|-i><rsup|<around*|(|k|)>>>\<delta\><rsub|i><around*|(|S|)>,>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|k=0><rsup|n-1>w<rsub|bzf>*<big|sum><rsub|S\<subseteq\>D<rsub|-i><rsup|<around*|(|k|)>>>\<delta\><rsub|i><around*|(|S|)>,<text|
    if >w<rsub|per>=1/w<rsub|sh>>>>>
  </eqnarray*>

  so we see that we must have <math|w<rsub|per><around*|(|k|)>=1/w<rsub|sh><around*|(|k|)>>
  in order to recover <math|v<rsub|bzf><around*|(|i|)>>.

  For uniform sampling of the powerset <math|S\<sim\>\<cal-U\><around*|(|D<rsub|-i>|)>>,
  we do as above, and, because <math|w<rsub|unif>*w<rsub|bzf>=1>:

  <\eqnarray*>
    <tformat|<table|<row|<cell|<wide|v|^><rsub|bzf,unif><around*|(|i|)>>|<cell|\<assign\>>|<cell|<frac|1|m>*<big|sum><rsub|j=1><rsup|m>w<rsub|unif>*w<rsub|bzf>*\<delta\><rsub|i><around*|(|S<rsub|j>|)>>>|<row|<cell|>|<cell|\<longrightarrow\>>|<cell|<below|\<bbb-E\>|S\<sim\>\<cal-U\><around*|(|D<rsub|-i>|)>><around*|[|\<delta\><rsub|i><around*|(|S|)>|]>>>|<row|<cell|>|<cell|=>|<cell|v<rsub|bzf><around*|(|i|)>.>>>>
  </eqnarray*>

  So we have a general way of mixing sampling strategies and semi-value
  coefficients. The drawback is that a direct implementation with that much
  cancelling of coefficients might be inefficient or numerically unstable. On
  the flip side, we can implement any sampling method, like antithetic
  sampling, and immediately benefit in all semi-value computations.
</body>

<\initial>
  <\collection>
    <associate|info-flag|detailed>
    <associate|prog-scripts|python>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
    <associate|footnote-1|<tuple|1|?>>
    <associate|footnr-1|<tuple|1|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <with|par-left|<quote|1tab>|1<space|2spc>Mixing sampling strategies for
      semi-values <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1>>
    </associate>
  </collection>
</auxiliary>