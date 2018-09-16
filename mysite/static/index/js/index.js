
function validPassword(){
    var password = $(input[name='password']).val();
    var confirmPassword = $(input[name="pwconfirmation"]).val();
    console.log(password);
    console.log(confirmPassword);
    if(password !== confirmPassword){
           alert("Password is not equal to comfirm password");
           return false;
     }
     return true;
 }
function submitSignin(){
     console.log("sign in");
     // first valid password;
     validPassword();
 }



 // $("#loginbutton").click(function (event) {
 //     var formData = new FormData(document.getElementById("loginform"));
 //     $.ajax({
 //         url:'/adrop_main/uploadimage/',
 //         type:'POST',
 //         cache:false,
 //         data:formData,
 //         processData:false,
 //         contentType:false,
 //         crossDomain: true,
 //
 //        success: function(data){
 //            console.log("success")
 //            console.log(data)
 //        },
 //        failure: function(data){
 //            console.log("failure");
 //            console.log(data.test_value);
 //        },
 //    });
 // });

