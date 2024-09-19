---
layout: post
title:  "MLIR Forward Pass to Simplify Bitshifting"
date:   2024-09-16 20:49:43 -0500
categories: jekyll update
---

[MLIR][mlir] is a compiler framework used to build reuseable infrastructure that targets heterogeneous hardware. It is an intermediate representation (IR) used to transform some input code into an optimized backend code that can run on some specialized hardware. 

Suppose you wanted to write an MLIR program that takes as input some code with two left bit shifts and combines them into a single shift. For example, the input might be a program like
```
module  {
  func.func @main(%arg0: i32, %arg1: i32) -> i32 {
    %c4_i32 = arith.constant 8 : i32
    %c16_i32 = arith.constant 12 : i32
    %0 = arith.shli %arg0, %c4_i32 : i32
    %1 = arith.shli %0, %c16_i32 : i32
    return %1 : i32
  }
}
```
We would expect the following output
```
func.func @main(%arg0: i32, %arg1: i32) -> i32 {
  %c20_i32 = arith.constant 20 : i32
  %0 = arith.shli %arg0, %c20_i32 : i32
  return %0 : i32
}
```
The MLIR program to do this might look like
{% highlight C++ %}
namespace mlir {
struct MultiToShiftPattern: public OpRewritePattern<arith::ShLIOp> {
    MultiToShiftPattern(mlir::MLIRContext *context) : OpRewritePattern<arith::ShLIOp> (context, 1) {}

    LogicalResult matchAndRewrite(arith::ShLIOp op, PatternRewriter &rewriter) const override {
        Value lhs = op.getOperand(0);
        Value rhs = op.getOperand(1);

        auto rhsDefiningOp = rhs.getDefiningOp<arith::ConstantIntOp>();
        int64_t rhs_value = rhsDefiningOp.value();

        // Extract the parameters from the first (nested) operation
        // (x << y) << z
        // lhsOp = (x << y)
        // llhs = x, lrhs = y
        auto lhsOp = lhs.getDefiningOp<arith::ShLIOp>();
        Value llhs = lhsOp.getOperand(0);
        Value lrhs = lhsOp.getOperand(1);
        auto lrhsDefiningOp = lrhs.getDefiningOp<arith::ConstantIntOp>();
        int64_t lrhs_value = lrhsDefiningOp.value();
        int32_t total_shift = rhs_value + lrhs_value;

        Value const_shift = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIntegerAttr(rewriter.getI32Type(), total_shift));
        arith::ShLIOp newShift = rewriter.create<arith::ShLIOp>(op.getLoc(), llhs, const_shift);
        rewriter.replaceOp(op, {newShift});

        return LogicalResult::success(true);
    }
};
} // namespate mlir
{% endhighlight %}

Then you can execute the function as follows:
{% highlight C++ %}
class MultiToShiftPass : public PassWrapper<MultiToShiftPass, OperationPass<func::FuncOp>> {
    StringRef getArgument() const final {
        return "instcombine";
    }

    StringRef getDescription() const final {
        return "A simple pass to combine any consecutive constant shifts left into a single shift left";
    }

    void runOnOperation() {
        // Gets the entire MLIR program
        func::FuncOp func_op = getOperation();
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<MultiToShiftPattern>(&getContext());

        mlir::GreedyRewriteConfig config;
        config.maxIterations = 1;
        config.maxNumRewrites = 1;
        auto rewrite_result = applyPatternsAndFoldGreedily(func_op, std::move(patterns), config);
    }
};
{% endhighlight %}

Some notes: `matchAndRewrite` in `MultiToShiftPattern` takes the operations in nested fashion, and the `GreedyRewriteConfig` controls the number of iterations and rewrites.

[mlir]: https://mlir.llvm.org
