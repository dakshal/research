function neuralnet_practice()
    x=[0 0 1;0 1 1;1 0 1;1 1 1];
    y=[0;0;1;1]

    syn0=2*rand(3,1)-1
    %syn1=2*rand(4,1)-1;

    for i=1:10000
    %    l1=inv(1+exp(-(x*syn0)));
    %    l2=(1+exp(-(l1*syn1)));
    %    l2_delta=(y-l2)*(l2*(1-l2));
    %    l1_delta=l2_delta.*(syn1)'*(l1*(1-l1));
    %    syn1=syn1+(l1)'.*l2_delta;
    %    syn0=syn0+x'.*l1_delta;
    %
        l0=x;
        l1=nonlin(dot(l0,syn0),0);
        l1_err=y-l1;
        l1_delta=cross(l1_err,nonlin(l1,1));
        syn0=syn0+dot(l0,l1_delta);
    end

end
function y = nonlin(x,deriv)
    if (deriv==1)
        y = cross(x,(1-x));
    end
    if(deriv==0)
        y = 1/(1+exp(-x));
    end
end
    