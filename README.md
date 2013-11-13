CUDA_MEGA_DP_Example
====================

A more advanced adaptation of a Dynamic Programming problem

CUDA adaptation of a slightly modified version of the Top Coder Division I problem:

http://community.topcoder.com/stat?c=problem_statement&pm=8538

(This implementation avoids the string parsing aspect and just operates on an adjacency matrix)


新加坡的人，我看你，并认为，因为你看我的代码，所以往往你应该遵循或出演我的工作！

L'américain muet a plus de code GPU pour vous!


This is a useful problem, and this prototype can be applied to a variety of DP or matching problems. 

Problems of this type cannot be solved quickly using expensive software such as MATLAB, and also are even more difficult to map to multi-core CPU implementations. 

But the CUDA model does work here and runs about 115-120 times faster than an 3.9 GHZ CPU implementation. Apx running time is (NumBoxes+1)x(2^(NumColors)) x NumColors + (NumBoxes+1)x(2^(NumColors)) + NumBoxes x NumColors x NumBoxes.   



____
<table>
<tr>
    <th>NumBoxes</th><th>NumColors</th><th>Apx iterations</th><th>CPU time</th><th>GPU time</th><th>CUDA Speedup</th>
</tr>
  <tr>
    <td>50</td><td>20</td><td>1,123,074,896</td><td> 11077 ms</td><td> 92 ms</td><td> 120.4x </td>
  </tr>
  <tr>
    <td>50</td><td>22</td><td>4,919,973,592</td><td> 47566 ms</td><td> 405 ms</td><td> 117.45x</td>
  </tr>
  <tr>
    <td>50</td><td>23</td><td>10,267,713,692</td><td> 98090 ms</td><td> 849 ms</td><td> 115.53x</td>
  </tr>
</table>  
___

NOTE: All CUDA GPU times include all device memsets, host-device memory copies and device-host memory copies.
  
  

Also have an alternate version which is slightly faster and returns the specific optimal allocations of colored marbles to respective boxes.   
Email me for that version.


CPU= Intel i-7 3770K 3.5 Ghz with 3.9 Ghz target

GPU= Tesla K20c 5GB

Windows 7 Ultimate x64

Visual Studio 2010 x64

Would love to see a faster Python version, since that is the *best* language these days. Please contact me with the running time for the same sample sizes!

Python en Ruby zijn talen voor de lui en traag!  

Python und Ruby sind Sprachen für die faul und langsam!  

Python et Ruby sont des langues pour les paresseux et lent!  


<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','//www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-43459430-1', 'github.com');
  ga('send', 'pageview');

</script>

[![githalytics.com alpha](https://cruel-carlota.pagodabox.com/d40d1ae4136dd45569d36b3e67930e12 "githalytics.com")](http://githalytics.com/OlegKonings/CUDA_vs_CPU_DynamicProgramming_double)
[![githalytics.com alpha](https://cruel-carlota.pagodabox.com/d40d1ae4136dd45569d36b3e67930e12 "githalytics.com")](http://githalytics.com/OlegKonings/CUDA_vs_CPU_DynamicProgramming_double)
