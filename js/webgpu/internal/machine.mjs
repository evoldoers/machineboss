/**
 * Machine, MachineState, MachineTransition classes for WFST representation.
 *
 * Port of python/machineboss/machine.py. Preserves weight expressions
 * without evaluating them — use machine-prep.mjs for numerical evaluation.
 *
 * Run: import { Machine } from './machine.mjs';
 */

import { readFileSync } from 'fs';

/**
 * A single transition in a WFST.
 */
export class MachineTransition {
  /**
   * @param {number} dest - Destination state index
   * @param {*} [weight=1] - Weight expression (number, string, or object)
   * @param {string|null} [input=null] - Input symbol (null = epsilon)
   * @param {string|null} [output=null] - Output symbol (null = epsilon)
   */
  constructor(dest, weight = 1, input = null, output = null) {
    this.dest = dest;
    this.weight = weight;
    this.input = input;
    this.output = output;
  }

  static fromJSON(j) {
    return new MachineTransition(
      j.to,
      j.weight !== undefined ? j.weight : 1,
      j.in || null,
      j.out || null,
    );
  }

  toJSON() {
    const d = { to: this.dest };
    if (this.input) d.in = this.input;
    if (this.output) d.out = this.output;
    if (this.weight !== 1) d.weight = this.weight;
    return d;
  }

  get isSilent() {
    return !this.input && !this.output;
  }
}

/**
 * A single state in a WFST.
 */
export class MachineState {
  /**
   * @param {MachineTransition[]} [trans=[]] - Outgoing transitions
   * @param {*} [name=null] - State name/id
   */
  constructor(trans = [], name = null) {
    this.trans = trans;
    this.name = name;
  }

  static fromJSON(j) {
    return new MachineState(
      (j.trans || []).map(t => MachineTransition.fromJSON(t)),
      j.id !== undefined ? j.id : null,
    );
  }

  toJSON() {
    const d = {};
    if (this.name !== null && this.name !== undefined) d.id = this.name;
    d.trans = this.trans.map(t => t.toJSON());
    return d;
  }
}

/**
 * A weighted finite-state transducer.
 */
export class Machine {
  /**
   * @param {MachineState[]} [state=[]] - States
   * @param {Object} [defs={}] - Parameter/function definitions
   */
  constructor(state = [], defs = {}) {
    this.state = state;
    this.defs = defs;
  }

  static fromJSON(j) {
    if (typeof j === 'string') j = JSON.parse(j);
    const m = new Machine(
      j.state.map(s => MachineState.fromJSON(s)),
      j.defs || {},
    );
    m._resolveStateNames();
    return m;
  }

  static fromFile(path) {
    return Machine.fromJSON(JSON.parse(readFileSync(path, 'utf8')));
  }

  _resolveStateNames() {
    const nameToIdx = {};
    for (let i = 0; i < this.state.length; i++) {
      const name = this.state[i].name;
      if (name !== null && name !== undefined) {
        const key = Array.isArray(name) ? JSON.stringify(name) : name;
        nameToIdx[key] = i;
      }
      nameToIdx[i] = i;
    }
    for (const s of this.state) {
      for (const t of s.trans) {
        if (typeof t.dest !== 'number') {
          const key = Array.isArray(t.dest) ? JSON.stringify(t.dest) : t.dest;
          if (key in nameToIdx) {
            t.dest = nameToIdx[key];
          } else {
            throw new Error(`Unknown state reference: ${t.dest}`);
          }
        }
      }
    }
  }

  toJSON() {
    const d = { state: this.state.map(s => s.toJSON()) };
    if (this.defs && Object.keys(this.defs).length > 0) d.defs = this.defs;
    return d;
  }

  toJSONString(indent) {
    return JSON.stringify(this.toJSON(), null, indent);
  }

  get nStates() {
    return this.state.length;
  }

  get startState() {
    return 0;
  }

  get endState() {
    return this.state.length - 1;
  }

  inputAlphabet() {
    const syms = new Set();
    for (const s of this.state) {
      for (const t of s.trans) {
        if (t.input) syms.add(t.input);
      }
    }
    return Array.from(syms).sort();
  }

  outputAlphabet() {
    const syms = new Set();
    for (const s of this.state) {
      for (const t of s.trans) {
        if (t.output) syms.add(t.output);
      }
    }
    return Array.from(syms).sort();
  }

  get nTransitions() {
    let n = 0;
    for (const s of this.state) n += s.trans.length;
    return n;
  }
}
