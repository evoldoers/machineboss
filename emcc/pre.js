// This code goes at the start of the generated boss JS
Module.noInitialRun = true;
Module.noExitRuntime = true;
Module.print = function (x) {
  if (Module.stdout)
    Module.stdout.push (x)
  else
    console.log (x)
}
