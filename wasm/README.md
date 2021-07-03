# Machine Boss

machineboss is an emscripten-compiled version of [Machine Boss](https://github.com/evoldoers/machineboss).

~~~~
var boss = require ('machineboss')
boss.runWithFiles ('-p bintern -p terndna')
    .then ((result) => console.log (result.stdout))
~~~~
