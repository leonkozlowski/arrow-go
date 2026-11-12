// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package compute

import (
	"context"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/compute/exec"
	"github.com/apache/arrow-go/v18/arrow/compute/internal/kernels"
)

// MatchSubstringOptions controls the substring matching functions such as
// starts_with, which match Pattern literally, not as a regular
// expression. With IgnoreCase, string and string_view input match Unicode
// simple case folds and require a valid UTF-8 pattern, while binary and
// binary_view input match ASCII case folds only. String case folding uses
// the tables of the Go toolchain's unicode package, so results can differ
// between Go versions.
type MatchSubstringOptions = kernels.MatchSubstringOptions

var startsWithDoc = FunctionDoc{
	Summary: "Check if strings start with a literal pattern",
	Description: "For each string in \"strings\", emit true iff it starts with a\n" +
		"given pattern. The pattern must be given in MatchSubstringOptions.\n" +
		"Null inputs emit null. Dictionary inputs are decoded to their value\n" +
		"type, and extension inputs use their storage type.",
	ArgNames:        []string{"strings"},
	OptionsType:     "MatchSubstringOptions",
	OptionsRequired: true,
}

var utf8LengthDoc = FunctionDoc{
	Summary: "Compute UTF8 string lengths",
	Description: "For each string in \"strings\", emit its length in UTF8 characters.\n" +
		"Null values emit null. Dictionary inputs are decoded to their value\n" +
		"type, and extension inputs use their storage type.",
	ArgNames: []string{"strings"},
}

// stringFunc is the function type shared by the string functions. Before
// picking a kernel, it copies string_view input that spans several data
// buffers into one buffer, and unwraps dictionary and extension input.
type stringFunc struct {
	ScalarFunction
}

func (fn *stringFunc) Execute(ctx context.Context, opts FunctionOptions, args ...Datum) (Datum, error) {
	if len(args) == 1 {
		coalesced, err := coalesceMultiBufferViewDatum(ctx, args[0])
		if err != nil {
			return nil, err
		}
		if coalesced != nil {
			defer coalesced.Release()
			args = []Datum{coalesced}
		}
	}
	return execInternal(ctx, fn, opts, -1, args...)
}

func (fn *stringFunc) DispatchBest(vals ...arrow.DataType) (exec.Kernel, error) {
	ensureDictionaryDecoded(vals...)
	ensureNoExtensionType(vals...)
	return fn.DispatchExact(vals...)
}

// RegisterScalarString adds the starts_with and utf8_length functions
// to reg.
func RegisterScalarString(reg FunctionRegistry) {
	for _, f := range []struct {
		name    string
		doc     FunctionDoc
		kernels []exec.ScalarKernel
	}{
		{"starts_with", startsWithDoc, kernels.GetStartsWithKernels()},
		{"utf8_length", utf8LengthDoc, kernels.GetUtf8LengthKernels()},
	} {
		fn := &stringFunc{*NewScalarFunction(f.name, Unary(), f.doc)}
		for _, k := range f.kernels {
			if err := fn.AddKernel(k); err != nil {
				panic(err)
			}
		}
		reg.AddFunction(fn, false)
	}
}

// StartsWith returns, for each value in input, whether it starts with
// opts.Pattern. Null values produce null. Dictionary inputs are decoded
// to their value type, and extension inputs use their storage type.
func StartsWith(ctx context.Context, opts MatchSubstringOptions, input Datum) (Datum, error) {
	return CallFunction(ctx, "starts_with", &opts, input)
}

// Utf8Length returns the number of characters in each string of input.
// Like Arrow C++, it doesn't check that the input is valid UTF-8: every
// byte that isn't a continuation byte counts as one character. The result
// is int64 for large_utf8 input and int32 otherwise, and nulls stay null.
func Utf8Length(ctx context.Context, input Datum) (Datum, error) {
	return CallFunction(ctx, "utf8_length", nil, input)
}
