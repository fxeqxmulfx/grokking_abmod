import Grokking
import Lean.Util.CollectAxioms

open Lean Elab Command in
run_cmd do
  let env ← getEnv
  let allowed := #[``propext, ``Classical.choice, ``Quot.sound]
  let mut count : Nat := 0
  for (name, info) in env.constants.toList do
    if (`Grokking).isPrefixOf name then
      match info with
      | .thmInfo _ =>
        let axioms ← Lean.collectAxioms name
        for ax in axioms do
          unless allowed.contains ax do
            throwError "{name} depends on the unapproved axiom {ax}"
        count := count + 1
      | _ => pure ()
  logInfo m!"Audited {count} Grokking theorems: only propext, Classical.choice, Quot.sound."
