#![allow(dead_code)]

#[cfg(test)]
use crate::benchmark;

use super::{handler, Bits, Context, Outcome};

#[derive(Copy, Clone)]
pub struct Register(usize);

/// A closure based expression.
pub struct Expr {
    handler: Box<dyn Fn(&mut Context, &mut Bits) -> Bits>,
}

pub enum Input {
    Register(Register),
    Immediate(Bits),
    Expr(Expr),
}

impl From<Register> for Input {
    fn from(register: Register) -> Self {
        Self::Register(register)
    }
}

impl From<Bits> for Input {
    fn from(bits: Bits) -> Self {
        Self::Immediate(bits)
    }
}

impl From<Expr> for Input {
    fn from(expr: Expr) -> Self {
        Self::Expr(expr)
    }
}

fn set_reg(context: &mut Context, reg: Register, new_value: Bits, reg0: &mut Bits) {
  if reg.0 == 0 {
    *reg0 = new_value;
  } else {
    context.set_reg(reg.0, new_value)
  }
}

impl Expr {
    /// Executes the given expression using the given [`Context`].
    pub fn execute(&self, context: &mut Context, reg0: &mut Bits) -> Bits {
        (self.handler)(context, reg0)
    }

    /// Creates a new [`Inst`] from the given closure.
    fn new<T>(handler: T) -> Self
    where
        T: Fn(&mut Context, &mut Bits) -> Bits + 'static,
    {
        Self {
            handler: Box::new(handler),
        }
    }

    pub fn add<P0, P1>(result: Register, lhs: P0, rhs: P1) -> Self
    where
        P0: Eval + 'static,
        P1: Eval + 'static,
    {
        Self::new(move |context, reg0| {
            let lhs = lhs.eval(context, reg0);
            let rhs = rhs.eval(context, reg0);
            let new_value = lhs.wrapping_add(rhs);
            set_reg(context, result, new_value, reg0);
            new_value
        })
    }

    pub fn sub<P0, P1>(result: Register, lhs: P0, rhs: P1) -> Self
    where
        P0: Eval + 'static,
        P1: Eval + 'static,
    {
        Self::new(move |context, reg0| {
            let lhs = lhs.eval(context, reg0);
            let rhs = rhs.eval(context, reg0);
            let new_value = lhs.wrapping_sub(rhs);
            set_reg(context, result, new_value, reg0);
            new_value
        })
    }

    pub fn mul<P0, P1>(result: Register, lhs: P0, rhs: P1) -> Self
    where
        P0: Eval + 'static,
        P1: Eval + 'static,
    {
        Self::new(move |context, reg0| {
            let lhs = lhs.eval(context, reg0);
            let rhs = rhs.eval(context, reg0);
            let new_value = lhs.wrapping_mul(rhs);
            set_reg(context, result, new_value, reg0);
            new_value
        })
    }
}

/// A closure based instruction.
pub struct Inst {
    /// The closure stores everything required for the instruction execution.
    handler: Box<dyn Fn(&mut Context, &mut Bits) -> Outcome>,
}

pub trait Eval {
    fn eval(&self, context: &mut Context, reg0: &mut Bits) -> Bits;
}

impl Eval for Register {
    fn eval(&self, context: &mut Context, reg0: &mut Bits) -> Bits {
        if self.0 == 0 {
          *reg0
        } else {
          context.get_reg(self.0)
        }
    }
}

impl Eval for Bits {
    fn eval(&self, _context: &mut Context, _reg0: &mut Bits) -> Bits {
        *self
    }
}

impl Eval for Expr {
    fn eval(&self, context: &mut Context, reg0: &mut Bits) -> Bits {
        self.execute(context, reg0)
    }
}

impl Inst {
    /// Executes the given instruction using the given [`Context`].
    pub fn execute(&self, context: &mut Context, reg0: &mut Bits) -> Outcome {
        (self.handler)(context, reg0)
    }

    /// Creates a new [`Inst`] from the given closure.
    fn new<T>(handler: T) -> Self
    where
        T: Fn(&mut Context, &mut Bits) -> Outcome + 'static,
    {
        Self {
            handler: Box::new(handler),
        }
    }

    pub fn exec(expr: Expr) -> Self {
        Self::new(move |context, reg0| {
            expr.eval(context, reg0);
            Outcome::Continue
        })
    }

    pub fn local_set<I>(result: Register, input: I) -> Self
    where
        I: Eval + 'static,
    {
        Self::new(move |context, reg0| {
            let new_value = input.eval(context, reg0);
            set_reg(context, result, new_value, reg0);
            Outcome::Continue
        })
    }

    /// Branches to the instruction indexed by `target` if the contents of `condition` are zero.
    pub fn branch_eqz(condition: Expr) -> Self {
        Self::new(move |context, reg0| {
            let condition = condition.execute(context, reg0);
            if condition == 0 {
                Outcome::Return
            } else {
                Outcome::Continue
            }
        })
    }

    /// Executes all the instructions in the basic block one after another.
    pub fn basic_block<I>(insts: I) -> Self
    where
        I: IntoIterator<Item = Inst>,
    {
        let insts = insts.into_iter().collect::<Vec<_>>().into_boxed_slice();
        Self::new(move |context, reg0| {
            for inst in &insts[..] {
                match inst.execute(context, reg0) {
                    Outcome::Continue => (),
                    Outcome::Return => return Outcome::Return,
                }
            }
            Outcome::Continue
        })
    }

    /// Loops the body until it returns.
    pub fn loop_block(body: Inst) -> Self {
        Self::new(move |context, reg0| loop {
            match body.execute(context, reg0) {
                Outcome::Continue => (),
                Outcome::Return => return Outcome::Return,
            }
        })
    }

    /// Returns execution of the function and returns the result in `result`.
    pub fn ret(result: Register) -> Self {
        Self::new(move |context, _reg0| handler::ret(context, result.0))
    }
}

#[test]
fn counter_loop() {
    let repetitions = 100_000_000;
    let inst = Inst::basic_block(vec![
        Inst::exec(Expr::add(Register(0), Register(0), repetitions)),
        Inst::loop_block(Inst::branch_eqz(Expr::sub(Register(0), Register(0), 1))),
    ]);
    let mut context = Context::default();
    let mut reg0 = 0u64;
    benchmark(|| inst.execute(&mut context, &mut reg0));
}
